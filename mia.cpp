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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>

using namespace cv;

CvMat *vis_img = NULL;
int att_size = -1;
int y_pad = 1;
int n_act_maps = 1;
int vis_rows = -1;
int vis_cols = -1;
bool draw = true;

extern char *vocab_ext;
extern int vocab_ext_token_size;

// struct ggml_tensor * wq = NULL;
// struct ggml_tensor * wk = NULL;
// struct ggml_tensor * wv = NULL;

struct ggml_tensor * output_norm = NULL;
struct ggml_tensor * output = NULL;
// struct ggml_tensor * token_embd = NULL;
// struct ggml_tensor * attn_output = NULL;

inline float v2(struct ggml_tensor *t, uint32_t y, uint32_t x) {
    return *(float *) ((char *) t->data + y*t->nb[1] + x*t->nb[0]);
}
inline float v3(struct ggml_tensor *t, uint32_t z, uint32_t y, uint32_t x) {
    return *(float *) ((char *) t->data + z*t->nb[2] + y*t->nb[1] + x*t->nb[0]);
}

int select_layer = -1;
int select_index = -1;

void unembed(struct ggml_tensor *t, int set_y);
void draw_px(int ix, int iy, float v, float v_scale, CvMat *vis_img);
void draw_px2(int ix, int iy, float v, float v_scale, CvMat *vis_img);
void apply_colormap(char *name, uint8_t *src_img, uint8_t *dst_img, int rows, int cols);
static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data);
struct ggml_tensor * get_float_weights(struct ggml_context * ctx0, struct ggml_cgraph * cgraph, char * name);

extern "C" void tensor_process_callback(struct ggml_tensor * tensor) {

    struct ggml_tensor *t = tensor;
    struct ggml_tensor * src0 = t->src[0];
    struct ggml_tensor * src1 = t->src[1];

    int nx = t->ne[0];
    int ny = t->ne[1];
    int nz = t->ne[2];

    std::stringstream st(t->name);
    std::string name(t->name);
    int layer_num = 0;
    auto n = std::count(name.begin(), name.end(), '-');
    if (n == 1) {
        std::string ln;
        getline(st, ln, '-');
        getline(st, ln, '-');
        layer_num = std::stoi(ln);
    }

    // if (strncmp(t->name, "kq_soft_max", 11) == 0) {
    //     std::cout << "tensor_process: " << name << ", " << layer_num << " att_size " << att_size << " n " << nx << " " << ny << " " << nz << std::endl;
    // }

    std::cout << "tensor_process: " << name << ", " << layer_num << " att_size " << att_size << " n " << nx << " " << ny << " " << nz << std::endl;

    if (strncmp(t->name, "result_wo", 9) == 0 && layer_num == 16) {
        printf("\nunembed LN %d %s:\n", layer_num, t->name);
        for (int y = ny-1; y < ny; y++) {
            unembed(t, y);
        }
    }

    if (ggml_n_dims(t) == 3) {
        for (int z = 0; z < nz; z++) {

            if (strncmp(t->name, "kq_soft_max", 11) == 0) {

                // std::cout << "kq_soft_max: " << z << std::endl;

                if (draw) {
                    char buffer[25];
                    sprintf(buffer, "%d", layer_num);
                    CvFont font;
                    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, 8);
                    cvPutText(vis_img, buffer, cvPoint(
                            t->ne[0] * 32 + 20,
                            layer_num * (att_size * 1 + y_pad) + 10),
                            &font, cvScalarAll(128));
                }

                int head_i = (layer_num * 32 + z);
                bool do_ablate = false;

                // if (select_layer >= 0 && select_index >= 0) {
                //     if (z != select_index && layer_num == select_layer) {
                //         do_ablate = true;
                //     }
                // }

                // for (int i = 0; i < 1024; i++) {
                //     if (ablate_a[i] == head_i) {
                //         do_ablate = true;
                //     }
                // }

                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        float *vp = (float *) ((char *) t->data + z*t->nb[2] + y*t->nb[1] + x*t->nb[0]);
                        float v = *vp;
                        if (do_ablate) {
                            *vp = 0.0f;
                            v = 0;
                        }
                        int iy = y + layer_num * (att_size * 1 + y_pad);
                        int ix = x + nx*z;
                        draw_px2(ix, iy, v, 255.0f, vis_img);
                    }
                }
            }
        }
    }
}

extern "C" void init_callback(struct ggml_cgraph * cgraph) {

    uint32_t size = (
        // 4096ull*4097ull + 
        // 4096ull*4097ull +
        // 4096ull*4097ull +
        // 4096ull*4097ull +
        // 32000ull * 4097ull + 4097ull +
        32000ull * 4097ull + 4097ull
        ) * sizeof(float);
    uint8_t *buf = (uint8_t *)malloc(size);
    struct ggml_init_params params;
    params.mem_size   = size;
    params.mem_buffer = buf;
    params.no_alloc   = false;
    struct ggml_context * ctx0 = ggml_init(params);

    // wq = get_float_weights(ctx0, cgraph, "blk.16.attn_q.weight"); 
    // wk = get_float_weights(ctx0, cgraph, "blk.16.attn_k.weight");
    // wv = get_float_weights(ctx0, cgraph, "blk.16.attn_v.weight"); 

    output_norm = get_float_weights(ctx0, cgraph, "output_norm.weight"); 
    output = get_float_weights(ctx0, cgraph, "output.weight");
    // token_embd = get_float_weights(ctx0, cgraph, "token_embd.weight");
    // attn_output = get_float_weights(ctx0, cgraph, "blk.16.attn_output.weight"); 
}

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

//=============================================================================
// hook into computation graph

    add_ggml_callback(ctx, tensor_process_callback);
    add_ggml_init_callback(ctx, init_callback);

//=============================================================================

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

//=============================================================================
// optional visualization
    if (draw) {
        att_size = embd_inp.size();
        vis_rows = 33 * (att_size * n_act_maps + y_pad);
        vis_cols = 32 * 32 + 200;
        vis_img = cvCreateMat(vis_rows, vis_cols, CV_8UC1);
        cvSetZero(vis_img);
    }
//=============================================================================

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

                if (draw && n_past == 0) {
                    char fname[128];
                    sprintf(fname, "/home/ubuntu/tmp/llama_vis.png");
                    CvMat *map_img = cvCreateMat(vis_rows, vis_cols, CV_8UC3);
                    apply_colormap(fname, vis_img->data.ptr, map_img->data.ptr, vis_rows, vis_cols);
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

void draw_px(int ix, int iy, float v, float v_scale, CvMat *vis_img) {
    if (ix < vis_img->cols && iy < vis_img->rows) {
        int v1 = 128 + v * v_scale;
        if (v1 > 255) v1 = 255;
        if (v1 < 0) v1 = 0;
        if (v1 > CV_MAT_ELEM(*vis_img, uchar, iy, ix)) {
            CV_MAT_ELEM(*vis_img, uchar, iy, ix) = v1;                         
        }
    } 
}

void draw_px2(int ix, int iy, float v, float v_scale, CvMat *vis_img) {
    if (ix < vis_img->cols && iy < vis_img->rows) {
        int v1 = abs(v * v_scale);
        if (v1 > 255) v1 = 255;
        if (v1 < 0) v1 = 0;
        if (v1 > CV_MAT_ELEM(*vis_img, uchar, iy, ix)) {
            CV_MAT_ELEM(*vis_img, uchar, iy, ix) = v1;                         
        }
    } 
}

void apply_colormap(char *name, uint8_t *src_img, uint8_t *dst_img, int rows, int cols) {
    cv::Mat in(rows, cols, CV_8UC1, src_img);
    cv::Mat out(rows, cols, CV_8UC3, dst_img);
    applyColorMap(in, out, cv::COLORMAP_INFERNO);
    std::string s(name);
    imwrite(s, out);
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

void unembed(struct ggml_tensor *t, int set_y) {

    const int nx = t->ne[0];
    float *rn = (float *)malloc(nx * sizeof(float));
    float *rf = (float *)malloc(output->ne[1] * sizeof(float));
    float *p = (float *)malloc(output->ne[1] * sizeof(float));

    for (int x = 0; x < nx; x++) {
        rn[x] = v2(t, set_y, x) * v2(output_norm, 0, x);
    }

    for (int y = 0; y < output->ne[1]; y++) {
        float dot = 0;
        for (int x = 0; x < nx; x++) {
            dot += rn[x] * v2(output, y, x);
        }
        rf[y] = dot;
    }

    // softmax
    float max = -FLT_MAX;
    int max_i = -1;
    for (int y = 0; y < output->ne[1]; y++) {
        if (rf[y] > max) {
            max = rf[y];
            max_i = y;
        } 
    }
    float sum = 0;
    for (int y = 0; y < output->ne[1]; y++) {
        p[y] = expf(rf[y]-max);
        sum += p[y];
    }
    for (int y = 0; y < output->ne[1]; y++) {
        p[y] /= sum;
    }

    // top-k
    for (int j = 0; j < 5; j++) {

        float max = -FLT_MAX;
        int max_i = -1;
        for (int y = 0; y < output->ne[1]; y++) {
            if (rf[y] > max) {
                max = rf[y];
                max_i = y;
            } 
        }
        printf("%s %.2f – ", &vocab_ext[max_i*vocab_ext_token_size], max);

        rf[max_i] = -FLT_MAX;
    }
    printf("\n");
}
