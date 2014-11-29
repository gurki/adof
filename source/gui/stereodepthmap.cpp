#include "stereodepthmap.h"
#include "ui_stereodepthmap.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>

#include "../../common/core/logger.h"
#include "QClickablePixmap.h"

#include <QtGui/QImage>
#include <QtCore/QSize>

// ************************************************************************
//  GUI interaction
// ************************************************************************


////////////////////////////////////////////////////////////////////////////////
StereoDepthMap::StereoDepthMap(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StereoDepthMap)
{
    ui->setupUi(this);

    // Set up action signals and slots
    connect(ui->actionLoadIm1,      SIGNAL(triggered()), 
        this,                       SLOT(openImage1()));
    connect(ui->actionLoadIm2,      SIGNAL(triggered()), 
        this,                       SLOT(openImage2()));
    connect(ui->actionLoadDepth,    SIGNAL(triggered()), 
        this,                       SLOT(openDepthMap()));
    connect(ui->buttonLoad1,        SIGNAL(clicked()), 
        this,                       SLOT(openImage1()));
    connect(ui->buttonLoad2,        SIGNAL(clicked()), 
        this,                       SLOT(openImage2()));
    connect(ui->actionLoad_Example, SIGNAL(triggered()),
        this,                       SLOT(openExample()));

    connect(ui->pushSaveDepth,      SIGNAL(clicked()),
        this,                       SLOT(saveDepthMap()));
    connect(ui->actionSaveDepth,    SIGNAL(triggered()),
        this,                       SLOT(saveDepthMap()));
    connect(ui->pushSaveResult,     SIGNAL(clicked()),
        this,                       SLOT(saveResult()));
    connect(ui->actionSaveFocus,    SIGNAL(triggered()),
        this,                       SLOT(saveResult()));

    connect(ui->buttonQuadratic,    SIGNAL(clicked()), 
        this,                       SLOT(startQuadratic()));
    connect(ui->buttonTV,           SIGNAL(clicked()), 
        this,                       SLOT(startTV()));
    connect(ui->buttonHuber,        SIGNAL(clicked()), 
        this,                       SLOT(startHuber()));

    connect(ui->buttonStop,         SIGNAL(clicked()), 
        this,                       SLOT(setAbort()));
    connect(ui->buttonExit1,        SIGNAL(clicked()),
        this,                       SLOT(exitThis()));
    connect(ui->buttonExit2,        SIGNAL(clicked()),
        this,                       SLOT(exitThis()));
    connect(ui->menuExit,           SIGNAL(aboutToShow()),
        this,                       SLOT(exitThis()));
    connect(ui->pushReset,          SIGNAL(clicked()),
        this,                       SLOT(reset()));

    connect(ui->pushBlurAgain,      SIGNAL(clicked()),
        this,                       SLOT(updateFocusBlur()));
    connect(ui->spinBoxDiscrete,    SIGNAL(valueChanged(int)),
        this,                       SLOT(loadDiscreteGPU()));

    // Set up GUI
    ui->progressBar->setRange(0,500);
    ui->progressBar->setValue(0);

    ui->doubleSpinBox->setValue(0.1);

    ui->tabWidget->setCurrentIndex(1);
    this->show();
    ui->tabWidget->setCurrentIndex(0);

    // other set ups
    abort_ = false;
}


////////////////////////////////////////////////////////////////////////////////
StereoDepthMap::~StereoDepthMap() {
    delete ui;
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openImage1() 
{
    QString fileName = QFileDialog::getOpenFileName(
        this, tr("Open File"), "", tr("Files (*.png)")
    );

    openImage1(fileName);
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openImage1(QString fileName) 
{
    Logger logger("StereoDepthMap::openImage1");

    if(!fileName.isEmpty())
    {
        QImage image(fileName);

        if(image.isNull()) {
            QMessageBox::information(this,"Image Viewer","Error Displaying image");
            return;
        }

        // load image in GraphicViewer
        QGraphicsScene* scene1 = new QGraphicsScene();
        QGraphicsScene* scene2 = new QGraphicsScene();

        QClickablePixmap* item1 = new QClickablePixmap(image);
        QClickablePixmap* item2 = new QClickablePixmap(image, this);

        scene1->addItem(item1);
        scene2->addItem(item2);

        ui->viewImage1->setScene(scene1);
        ui->viewImage1->fitInView(item1, Qt::KeepAspectRatio);
        ui->viewImage1->show();

        ui->viewFocus->setScene(scene2);
        ui->viewFocus->fitInView(item2, Qt::KeepAspectRatio);
        ui->viewFocus->show();

        // initialize Diffuser and Inpainter for this image
        cv::Mat img = cv::imread(fileName.toStdString());
        lDiff_.setImage(img);
        inPaint_.setImage(img);
        dEstimator.setLeftImage(img);
    }
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openImage2(QString fileName)
{
    Logger logger("StereoDepthMap::openImage2");

    if(!fileName.isEmpty())
    {
        QImage image(fileName);

        if(image.isNull()) {
            QMessageBox::information(this,"Image Viewer","Error Displaying image");
            return;
        }

        // load image in GraphicViewer
        QGraphicsScene* scene = new QGraphicsScene();
        QClickablePixmap* item = new QClickablePixmap(image);
        scene->addItem(item);

        ui->viewImage2->setScene(scene);
        ui->viewImage2->fitInView(item, Qt::KeepAspectRatio);
        ui->viewImage2->show();

        //  set dispest image
        cv::Mat img = cv::imread(fileName.toStdString());
        dEstimator.setRightImage(img);
    }
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openImage2() {
    QString fileName = QFileDialog::getOpenFileName(
        this, tr("Open File"), "", tr("Files (*.png)")
    );

    openImage2(fileName);
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openDepthMap(QImage image) 
{
    Logger logger("StereoDepthMap::openDepthMap");

    if(image.isNull()) {
        QMessageBox::information(this, "Image Viewer", "Error Displaying image");
        return;
    }

    // load image in GraphicViewer
    QGraphicsScene* scene = new QGraphicsScene();
    QClickablePixmap* item = new QClickablePixmap(image);
    scene->addItem(item);

    ui->viewStereo1->setScene(scene);
    ui->viewStereo1->fitInView(item, Qt::KeepAspectRatio);
    ui->viewStereo1->show();

    ui->viewStereo2->setScene(scene);
    ui->viewStereo2->fitInView(item, Qt::KeepAspectRatio);
    ui->viewStereo2->show();
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::loadDepthMap(QImage image) 
{
    openDepthMap(image);

    lDiff_.setDisparityMap(depthMap_);
    inPaint_.setMask(depthMap_);
}

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openDepthMap(QString fileName)
{
    if(!fileName.isEmpty())
    {
        QImage image(fileName);

        openDepthMap(image);

        //  set diff disparity map
        depthMap_ = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
        lDiff_.setDisparityMap(depthMap_);
        inPaint_.setMask(depthMap_);
    }
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openDepthMap()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, tr("Open File"), "", tr("Files (*.png)")
    );

    openDepthMap(fileName);
}

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::saveDepthMap()
{
    Logger logger("StereoDepthMap::saveDepthMap()");
    if(!depthMap_.empty()) 
    {
        QString fileName = QFileDialog::getSaveFileName(
            this, tr("Save Depth Map"), "", tr("Files (*.png)")
        );

        cv::imwrite(fileName.toStdString(), depthMap_);
    } 
    else {
        logger << "No Depth Map exists."; logger.eol();
    }
}

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::saveResult()
{
    Logger logger("StereoDepthMap::saveResult()");
    if(!output_.isNull()) 
    {
        QString fileName = QFileDialog::getSaveFileName(
            this, tr("Save Result"), "", tr("Files (*.png)")
        );

        if (output_.save(fileName, "PNG", 100)) {
            logger << "Saved successfully."; logger.eol();
        } else {
            logger << "Could not save image."; logger.eol();
        }
    } else {
        logger << "No Result Image exists."; logger.eol();
    }
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::openExample() 
{
    Logger logger("StereoDepthMap::openExample");

    QString fileName1("../data/small/2005/Dolls/view1.png");
    openImage1(fileName1);

    QString fileName2("../data/small/2005/Dolls/view0.png");
    openImage2(fileName2);

    QString fileName3("../data/small/2005/Dolls/disp1.png");
    openDepthMap(fileName3);
}


////////////////////////////////////////////////////////////////////////////////
// Set all GraphicViews empty
void StereoDepthMap::reset()
{
	lDiff_.reset();
    inPaint_.reset();
    QImage image = getDiffusionQImage();
	
    // load image in GraphicViewer
    QGraphicsScene* scene = new QGraphicsScene();
    QClickablePixmap* item = new QClickablePixmap(image, this);
    scene->addItem(item);

    ui->viewFocus->setScene(scene);
    ui->viewFocus->fitInView(item, Qt::KeepAspectRatio);
    ui->viewFocus->show();
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::setAbort() {
    abort_ = true;
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::exitThis() {
    abort_ = true;
    QCoreApplication::exit();
}


// ************************************************************************
//  Interaction with Disparity Estimator
// ************************************************************************

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::updateFocusBlur() 
{ 
    int iterations = ui->spinBoxPowerBlur->value();

    for(int i=0; i <= iterations; i++) {
        lDiff_.adaptiveUpdate();
    }

    QImage image = getDiffusionQImage();

    loadImageInFocus(image);
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::loadImageInFocus(QImage image) {
    // load image in GraphicViewer
    QGraphicsScene* scene = new QGraphicsScene();
    QClickablePixmap* item = new QClickablePixmap(image, this);
    scene->addItem(item);

    // QGraphicsItem* item = (QGraphicsItem*)ui->viewFocus->scene()->items.first();
    // item->setPixmap(QPixmap::fromImage(image));

    ui->viewFocus->setScene(scene);
    ui->viewFocus->fitInView(item, Qt::KeepAspectRatio);
    ui->viewFocus->show();

    // remember image
    output_ = image;
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::loadDiscreteGPU() {
    // load image in GraphicViewer
    cv::Mat cv_dm = depthMap_;

    const int nlayer = ui->spinBoxDiscrete->value();

    if(nlayer > 0) {
        discretizeDepthMap(cv_dm, nlayer);
    }

    lDiff_.setDisparityMap(cv_dm);
    inPaint_.setMask(cv_dm);

    // load image in GraphicViewer
    cv_dm.convertTo(cv_dm, CV_8U);
    cv::cvtColor(cv_dm, cv_dm, CV_GRAY2RGB);

    QImage image((uchar*)cv_dm.ptr(), cv_dm.cols, cv_dm.rows, cv_dm.step, QImage::Format_RGB888);
    QImage dstCpy(image);
    dstCpy.detach();

    QGraphicsScene* scene = new QGraphicsScene();
    QClickablePixmap* item = new QClickablePixmap(dstCpy);
    scene->addItem(item);

    ui->viewStereo2->setScene(scene);
    ui->viewStereo2->fitInView(item, Qt::KeepAspectRatio);
    ui->viewStereo2->show();
}

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::discretizeDepthMap(cv::Mat& disc, int nlayer) {
    // discretize depth map
    double min, max;
    cv::minMaxIdx(depthMap_, &min, &max);

    depthMap_.copyTo(disc);

    disc.convertTo(disc, CV_32F);
    disc -= min;
    disc /= (max - min);
    disc *= nlayer;
    disc.convertTo(disc, CV_8U);
    disc *= (255.0f / nlayer);
}

////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::didPressButton(int x, int y) 
{
    Logger logger("StereoDepthMap::didPressButton");

    int ind = ui->tabWidget_2->currentIndex();

    // case focus blur
    if(ind == 0 && lDiff_.isValid()) 
    {
        // initialize linear diffusion
        lDiff_.reset();

        int iterations = ui->spinBoxPowerBlur->value();

        if(ui->radioLinear->isChecked())
            lDiff_.setEta(1);
        else 
            lDiff_.setEta(2);

        lDiff_.setFocus(x, y);

        Logger::setMaxLevel(1);
        
        for(int i=0; i <= iterations; i++) {
            lDiff_.adaptiveUpdate();
        }

        Logger::setMaxLevel(-1);

        output_ = getDiffusionQImage();
    } 
    // case inpainting
    else if (ind == 1 && inPaint_.isValid()) 
    { 
        inPaint_.reset();

        int iterations = ui->spinBoxInpainting->value();

        inPaint_.setFocus(x, y);

        Logger::setMaxLevel(1);

        for(int i=0; i <= iterations; i++) {
            inPaint_.update();
        }

        Logger::setMaxLevel(-1);

        output_ = getInpaintingQImage();
    }
    else if (ind == 2) 
    {
        // initialize highlighting by linear diff
        lDiff_.reset();

        lDiff_.setEta(ui->doubleSpinBox->value());

        lDiff_.setFocus(x, y);

        Logger::setMaxLevel(1);
        
        lDiff_.intensityUpdate();

        Logger::setMaxLevel(-1);

        output_ = getDiffusionQImage();
    }
    else {
        logger << "input image or disparity map missing"; logger.eol();
        logger.pop(false);
    }

    loadImageInFocus(output_);
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::didMoveMouse(int x, int y) 
{
    // cout << x << ", " << y << endl;

    // Logger logger("StereoDepthMap::didMoveMouse");

    // int ind = ui->tabWidget_2->currentIndex();

    // if (ind == 2 && lDiff_.isValid()) 
    // {
    //     // initialize linear diffusion
    //     lDiff_.reset();
    //     lDiff_.setFocus(x, y);

    //     int iterations = ui->spinBoxPowerBlur->value();

    //     Logger::setMaxLevel(1);
        
    //     lDiff_.intensityUpdate();

    //     Logger::setMaxLevel(-1);

    //     output_ = getDiffusionQImage();
    // }
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::readParameters() {
    dEstimator.setDepth(ui->spinBoxDepth->value());
    dEstimator.setAlpha(ui->spinBoxAlpha->value());
    dEstimator.setTheta(ui->spinBoxTheta->value());
    dEstimator.setMu(ui->spinBoxDataTerm->value());
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::startQuadratic() {
    StereoDepthMap::readParameters();

    dEstimator.setRegularizer("quadratic");
    dEstimator.initialize();

    calculateDepthMap();
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::startTV() {
    StereoDepthMap::readParameters();

    dEstimator.setRegularizer("tv");
    dEstimator.initialize();

    calculateDepthMap();
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::startHuber() {
    StereoDepthMap::readParameters();

    dEstimator.setRegularizer("huber");
    dEstimator.initialize();

    calculateDepthMap();
}

////////////////////////////////////////////////////////////////////////////////
QImage StereoDepthMap::getInpaintingQImage() {

    // // convert Mat to QImage
    cv::Mat dst, src;
    inPaint_.getInpaintedImage(src);
    cv::cvtColor(src, dst, CV_BGR2RGB);

    QImage image((uchar*)dst.ptr(), dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
    QImage dstCpy(image);
    dstCpy.detach();

    return dstCpy;
}

////////////////////////////////////////////////////////////////////////////////
QImage StereoDepthMap::getDiffusionQImage() {

    // // convert Mat to QImage
    cv::Mat dst, src;
    lDiff_.getDiffusedImage(src);
    cv::cvtColor(src, dst, CV_BGR2RGB);

    QImage image((uchar*)dst.ptr(), dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
    QImage dstCpy(image);
    dstCpy.detach();

    return dstCpy;
}


////////////////////////////////////////////////////////////////////////////////
QImage StereoDepthMap::getDepthQImage() {
    //QImage image = dEstimator.getDeviceDisparity();

    // // convert Mat to QImage
    cv::Mat dst, src;
    dEstimator.getDeviceDisparity(src);
    cv::cvtColor(src, dst, CV_GRAY2RGB);

    QImage image((uchar*)dst.ptr(), dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
    QImage dstCpy(image);
    dstCpy.detach();

    return dstCpy;
}


////////////////////////////////////////////////////////////////////////////////
void StereoDepthMap::calculateDepthMap() {
    //Logger logger("StereoDepthMap::calculateDepthMap");

    int key = 0;
    int iter = 0;
    abort_ = false;

    ui->progressBar->setValue(0);

    while (true)
    {
        if(iter <= 500)
            ui->progressBar->setValue(iter);
        iter++;

        //stringstream str; str << "iter: " << ++iter;
        //logger.push(str.str());

        dEstimator.update();
        
        // load Depth Map into Graphic Viewer

        // load image in GraphicViewer
            QGraphicsScene* scene = new QGraphicsScene();
            QImage image = StereoDepthMap::getDepthQImage();
            QClickablePixmap* item = 
                new QClickablePixmap(image, this);
            scene->addItem(item);

            ui->viewStereo1->setScene(scene);
            ui->viewStereo1->fitInView(item, Qt::KeepAspectRatio);
            ui->viewStereo1->show();
        
        // de.showDeviceGradient(4, REDUCE_XY, "gradXY");
        // de.showDeviceGradient(img1.cols / 2, REDUCE_YZ, "gradYZ");
        // de.showDeviceGradient(img1.rows / 2, REDUCE_ZX, "gradZX");

        //logger.pop();

        // may abort calculation out of the gui
        QCoreApplication::processEvents();

        if (abort_) {
            // load StereoDepthMap in Focus View
            dEstimator.getDeviceDisparity(depthMap_);

            loadDepthMap(image);

            break;
        }
    }

}
