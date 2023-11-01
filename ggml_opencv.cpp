#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

extern "C" void apply_colormap(char *name, uint8_t *src_img, uint8_t *dst_img, int rows, int cols);

void apply_colormap(char *name, uint8_t *src_img, uint8_t *dst_img, int rows, int cols) {
    cv::Mat in(rows, cols, CV_8UC1, src_img);
    cv::Mat out(rows, cols, CV_8UC3, dst_img);
    applyColorMap(in, out, COLORMAP_INFERNO);

    std::string s(name);

    std::cout << "apply_colormap " << s << std::endl;

    imwrite(s, out);
}

// enum ColormapTypes
// {
//     COLORMAP_AUTUMN = 0, //!< ![autumn](pics/colormaps/colorscale_autumn.jpg)
//     COLORMAP_BONE = 1, //!< ![bone](pics/colormaps/colorscale_bone.jpg)
//     COLORMAP_JET = 2, //!< ![jet](pics/colormaps/colorscale_jet.jpg)
//     COLORMAP_WINTER = 3, //!< ![winter](pics/colormaps/colorscale_winter.jpg)
//     COLORMAP_RAINBOW = 4, //!< ![rainbow](pics/colormaps/colorscale_rainbow.jpg)
//     COLORMAP_OCEAN = 5, //!< ![ocean](pics/colormaps/colorscale_ocean.jpg)
//     COLORMAP_SUMMER = 6, //!< ![summer](pics/colormaps/colorscale_summer.jpg)
//     COLORMAP_SPRING = 7, //!< ![spring](pics/colormaps/colorscale_spring.jpg)
//     COLORMAP_COOL = 8, //!< ![cool](pics/colormaps/colorscale_cool.jpg)
//     COLORMAP_HSV = 9, //!< ![HSV](pics/colormaps/colorscale_hsv.jpg)
//     COLORMAP_PINK = 10, //!< ![pink](pics/colormaps/colorscale_pink.jpg)
//     COLORMAP_HOT = 11, //!< ![hot](pics/colormaps/colorscale_hot.jpg)
//     COLORMAP_PARULA = 12 //!< ![parula](pics/colormaps/colorscale_parula.jpg)
// };

