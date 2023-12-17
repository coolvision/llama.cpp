#include "common.h"

#include "console.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

extern "C" void tensor_process(char *name, int layer_num) {


    printf("process %s %d\n", name, layer_num);



};

typedef void (*ggml_compute_callback)(char *name, int layer_num);


int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    llama_sampling_params & sparams = params.sparams;

    log_set_target(stdout);

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    // if (!params.verbose) {
    llama_log_set(llama_null_log_callback, NULL);
    // }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;
    llama_context * ctx_guidance = NULL;    

    // load the model
    LOG("%s: load the model\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    add_ggml_callback(ctx, tensor_process);

    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;

    LOG("tokenize the prompt\n");
    if (params.chatml) {
        params.prompt = "<|im_start|>system\n" + params.prompt + "<|im_end|>";
    }
    embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int) embd_inp.size(); i++) {
        LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
    }

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    bool is_antiprompt        = false;
    bool input_echo           = true;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<int>   input_tokens;
    std::vector<int>   output_tokens;
    std::ostringstream output_ss;

    std::vector<llama_token> embd;

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {

        // predict
        if (!embd.empty()) {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                // LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                // LOG("n_past = %d\n", n_past);
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
 
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            // LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            // LOG("n_remain: %d\n", n_remain);

        } else {
            // some user input remains from prompt or interaction, forward it to processing
            // LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo) {
            for (auto id : embd) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                printf("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                } else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
            fflush(stdout);
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(model) && !(params.instruct || params.interactive || params.chatml)) {
            LOG_TEE(" [end of text]\n");
            break;
        }
    }

    if (ctx_guidance) { llama_free(ctx_guidance); }
    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctx_sampling);
    llama_backend_free();

    return 0;
}
