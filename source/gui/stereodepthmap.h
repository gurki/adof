#ifndef STEREODEPTHMAP_H
#define STEREODEPTHMAP_H

#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QGraphicsPixmapItem>
#include <QtGui/QMouseEvent>

#include "ui_stereodepthmap.h"

#include "../disparity/DisparityEstimator.h"
#include "../diffusion/LinearDiffusion.h"
#include "../inpainting/Inpainting.h"

namespace Ui {
class StereoDepthMap;
}

class StereoDepthMap : public QMainWindow
{
    Q_OBJECT

public:
    explicit StereoDepthMap(QWidget *parent = 0);
    ~StereoDepthMap();

    void calculateDepthMap();
    void initializeEventFilter();
    void loadImagesInDE();
    void readParameters();

    QImage  getDepthQImage();
    QImage  getDiffusionQImage();
    QImage  getInpaintingQImage();

    void    didPressButton(int x, int y);
    void    didMoveMouse(int x, int y);

    void openImage1(QString filename);
    void openImage2(QString fileName);
    void openDepthMap(QImage image);
    void loadDepthMap(QImage image);
    void openDepthMap(QString fileName);
    void loadImageInFocus(QImage image);

    void discretizeDepthMap(cv::Mat& disc, int nlayer);


private slots:
    virtual void startQuadratic();
    virtual void startTV();
    virtual void startHuber();
    
    virtual void openImage1();
    virtual void openImage2();
    virtual void openDepthMap();
    virtual void openExample();
    virtual void saveDepthMap();
    virtual void saveResult();

    virtual void updateFocusBlur();
    virtual void loadDiscreteGPU();

    virtual void reset();
    virtual void setAbort();
    virtual void exitThis();


private:
    Ui::StereoDepthMap *ui;

    DisparityEstimator  dEstimator;
    Inpainting          inPaint_;
    LinearDiffusion     lDiff_;

    bool abort_;

    cv::Mat depthMap_;
    QImage output_;
};

#endif // STEREODEPTHMAP_H
