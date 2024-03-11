#include <emscripten/emscripten.h>
//#include <emscripten/val.h>
//#include <emscripten/bind.h>
#include <simpleocv.h>
#include <net.h>
#include <datareader.h>
#include <cpu.h>

#include "model.h"

ncnn::Net* retinaface = nullptr;

struct FaceObject {
    float rect[4];
    float landmark[5][2];
    float prob;
};

static inline float intersection_area(const FaceObject &a, const FaceObject &b) {
    // 没有 opencv
    float x0 = std::max(a.rect[0], b.rect[0]);
    float y0 = std::max(a.rect[1], b.rect[1]);
    float x1 = std::min(a.rect[2], b.rect[2]);
    float y1 = std::min(a.rect[3], b.rect[3]);

    float w = x1 - x0;
    float h = y1 - y0;
    if (w < 0 || h < 0)
        return 0.f;

    return w * h;
}

static void qsort_descent_inplace(std::vector<FaceObject> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject> &faceObjects) {
    if (faceObjects.empty()) return;

    qsort_descent_inplace(faceObjects, 0, (int) faceObjects.size() - 1);
}

static void nms_sorted_bboxes(
        const std::vector<FaceObject> &faceObjects, std::vector<int> &picked, float nms_threshold
) {
    picked.clear();

    const int n = (int) faceObjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        auto& rect = faceObjects[i].rect;
        areas[i] = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1);
    }

    for (int i = 0; i < n; i++) {
        const FaceObject &a = faceObjects[i];

        int keep = 1;
        for (int j: picked) {
            const FaceObject &b = faceObjects[j];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float *anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void
generate_proposals(const ncnn::Mat &anchors, float feat_stride, const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob,
                   const ncnn::Mat &landmark_blob, float prob_threshold, std::vector<FaceObject> &faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++) {
        const float *anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++) {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold) {
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    FaceObject obj = {
                            {x0, y0, x1 + 1, y1 + 1},
                            {
                                    {cx + (anchor_w + 1) * landmark.channel(0)[index], cy + (anchor_h + 1) * landmark.channel(1)[index]},
                                    {cx + (anchor_w + 1) * landmark.channel(2)[index], cy + (anchor_h + 1) * landmark.channel(3)[index]},
                                    {cx + (anchor_w + 1) * landmark.channel(4)[index], cy + (anchor_h + 1) * landmark.channel(5)[index]},
                                    {cx + (anchor_w + 1) * landmark.channel(6)[index], cy + (anchor_h + 1) * landmark.channel(7)[index]},
                                    {cx + (anchor_w + 1) * landmark.channel(8)[index], cy + (anchor_h + 1) * landmark.channel(9)[index]}
                            },
                            prob
                    };

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

static bool init_retinaface() {
    if (retinaface) return true;
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(1);

    retinaface = new ncnn::Net;
    retinaface->opt.num_threads = 1;
    if (retinaface->load_param_mem(FILES_mnet_25_params)) {
        delete retinaface;
        retinaface = nullptr;
        return false;
    }

    const unsigned char* g = FILES_mnet_25_opt_bin;
    ncnn::DataReaderFromMemory rd(g);

    if (retinaface->load_model(rd)) {
        delete retinaface;
        retinaface = nullptr;
        return false;
    }
    return true;
}

static int detect_retinaface(const unsigned char* data, int img_w, int img_h, std::vector<FaceObject> &faceobjects) {
    if (!init_retinaface()) return -1;

    const float prob_threshold = 0.75f;
    const float nms_threshold = 0.4f;

    ncnn::Mat in = ncnn::Mat::from_pixels(data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);

    ncnn::Extractor ex = retinaface->create_extractor();
    ex.set_light_mode(true);

    auto res = ex.input("data", in);
    if (res) return res;

    std::vector<FaceObject> faceproposals;

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    auto face_count = (int)picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++) {
        faceobjects[i] = faceproposals[picked[i]];

        // clip to image size
        float x0 = faceobjects[i].rect[0];
        float y0 = faceobjects[i].rect[1];
        float x1 = x0 + faceobjects[i].rect[2];
        float y1 = y0 + faceobjects[i].rect[3];

        x0 = std::max(std::min(x0, (float) img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float) img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float) img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float) img_h - 1), 0.f);

        faceobjects[i].rect[0] = x0;
        faceobjects[i].rect[1] = y0;
        faceobjects[i].rect[2] = x1;
        faceobjects[i].rect[3] = y1;
    }

    return 0;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE void* _malloc(size_t size) {
    return malloc(size);
}

EMSCRIPTEN_KEEPALIVE void _free(void* ptr) {
    free(ptr);
}

EMSCRIPTEN_KEEPALIVE void* detect(const unsigned char* data, int img_w, int img_h) {
    std::vector<FaceObject> faceobjects;
    auto res = detect_retinaface(data, img_w, img_h, faceobjects);
    if (res) return nullptr;

    auto size = (int)faceobjects.size();
    auto mem = (float*) malloc(4 * (1 + size * (4 + 5 * 2 + 1)));
    memccpy(mem, &size, 1, 4);
    auto resMem = mem + 1;
    for (auto& face : faceobjects) {
        for (int j = 0; j < 4; j++) {
            resMem[j] = face.rect[j];
        }
        for (int j = 0; j < 5; j++) {
            resMem[4 + j * 2] = face.landmark[j][0];
            resMem[4 + j * 2 + 1] = face.landmark[j][1];
        }
        resMem[4 + 5 * 2] = face.prob;
        resMem += 4 + 5 * 2 + 1;
    }
    return mem;
}
}
