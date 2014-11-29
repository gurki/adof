/********************************************************************************
** Form generated from reading UI file 'stereodepthmap.ui'
**
** Created by: Qt User Interface Compiler version 4.8.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_STEREODEPTHMAP_H
#define UI_STEREODEPTHMAP_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFormLayout>
#include <QtGui/QGraphicsView>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QProgressBar>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QTabWidget>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_StereoDepthMap
{
public:
    QAction *actionLoadIm1;
    QAction *actionLoadIm2;
    QAction *actionSaveDepth;
    QAction *actionSaveFocus;
    QAction *actionLoadDepth;
    QAction *actionLoad_Example;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout_11;
    QTabWidget *tabWidget;
    QWidget *tab;
    QHBoxLayout *horizontalLayout_12;
    QVBoxLayout *verticalLayout_10;
    QHBoxLayout *horizontalLayout_9;
    QVBoxLayout *verticalLayout_8;
    QHBoxLayout *horizontalLayout_7;
    QLabel *labelInput1;
    QPushButton *buttonLoad1;
    QGraphicsView *viewImage1;
    QVBoxLayout *verticalLayout_9;
    QHBoxLayout *horizontalLayout_8;
    QLabel *labelInput2;
    QPushButton *buttonLoad2;
    QGraphicsView *viewImage2;
    QFormLayout *formLayout;
    QLabel *labelDepth;
    QSpinBox *spinBoxDepth;
    QLabel *labelDataTerm;
    QLabel *labelAlpha;
    QDoubleSpinBox *spinBoxAlpha;
    QSpinBox *spinBoxDataTerm;
    QLabel *labelTheta;
    QDoubleSpinBox *spinBoxTheta;
    QSpacerItem *verticalSpacer_11;
    QSpacerItem *verticalSpacer_12;
    QSpacerItem *verticalSpacer_4;
    QHBoxLayout *horizontalLayout_10;
    QVBoxLayout *verticalLayout_4;
    QSpacerItem *verticalSpacer_10;
    QLabel *labelRegularizer;
    QSpacerItem *verticalSpacer_9;
    QPushButton *buttonTV;
    QPushButton *buttonQuadratic;
    QPushButton *buttonHuber;
    QSpacerItem *verticalSpacer_8;
    QPushButton *pushSaveDepth;
    QSpacerItem *horizontalSpacer_4;
    QVBoxLayout *verticalLayout_7;
    QGridLayout *gridLayout;
    QPushButton *buttonStop;
    QLabel *labelMap;
    QProgressBar *progressBar;
    QSpacerItem *horizontalSpacer;
    QGraphicsView *viewStereo1;
    QPushButton *buttonExit1;
    QWidget *tab_2;
    QHBoxLayout *horizontalLayout_13;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_5;
    QLabel *label;
    QHBoxLayout *horizontalLayout_5;
    QSpacerItem *horizontalSpacer_3;
    QGraphicsView *viewStereo2;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_9;
    QSpinBox *spinBoxDiscrete;
    QLabel *label_6;
    QSpacerItem *verticalSpacer_6;
    QLabel *label_5;
    QSpacerItem *verticalSpacer_3;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_2;
    QGraphicsView *viewFocus;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_6;
    QSpacerItem *verticalSpacer_5;
    QPushButton *pushSaveResult;
    QPushButton *pushReset;
    QPushButton *buttonExit2;
    QSpacerItem *horizontalSpacer_2;
    QTabWidget *tabWidget_2;
    QWidget *tab_3;
    QWidget *formLayoutWidget_2;
    QFormLayout *formLayout_3;
    QRadioButton *radioLinear;
    QRadioButton *radioQuadratic;
    QSpinBox *spinBoxPowerBlur;
    QLabel *label_3;
    QPushButton *pushBlurAgain;
    QSpacerItem *verticalSpacer;
    QSpacerItem *verticalSpacer_7;
    QWidget *tab_5;
    QSpinBox *spinBoxInpainting;
    QLabel *label_4;
    QLabel *label_7;
    QLabel *label_8;
    QWidget *tab_4;
    QWidget *formLayoutWidget_4;
    QFormLayout *formLayout_5;
    QDoubleSpinBox *doubleSpinBox;
    QLabel *label_10;
    QHBoxLayout *horizontalLayout_3;
    QHBoxLayout *horizontalLayout;
    QMenuBar *menuBar;
    QMenu *menuLoad_Images;
    QMenu *menuSave;
    QMenu *menuExit;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *StereoDepthMap)
    {
        if (StereoDepthMap->objectName().isEmpty())
            StereoDepthMap->setObjectName(QString::fromUtf8("StereoDepthMap"));
        StereoDepthMap->resize(866, 722);
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(StereoDepthMap->sizePolicy().hasHeightForWidth());
        StereoDepthMap->setSizePolicy(sizePolicy);
        StereoDepthMap->setAcceptDrops(true);
        StereoDepthMap->setAnimated(true);
        StereoDepthMap->setDocumentMode(true);
        actionLoadIm1 = new QAction(StereoDepthMap);
        actionLoadIm1->setObjectName(QString::fromUtf8("actionLoadIm1"));
        actionLoadIm2 = new QAction(StereoDepthMap);
        actionLoadIm2->setObjectName(QString::fromUtf8("actionLoadIm2"));
        actionSaveDepth = new QAction(StereoDepthMap);
        actionSaveDepth->setObjectName(QString::fromUtf8("actionSaveDepth"));
        actionSaveFocus = new QAction(StereoDepthMap);
        actionSaveFocus->setObjectName(QString::fromUtf8("actionSaveFocus"));
        actionLoadDepth = new QAction(StereoDepthMap);
        actionLoadDepth->setObjectName(QString::fromUtf8("actionLoadDepth"));
        actionLoad_Example = new QAction(StereoDepthMap);
        actionLoad_Example->setObjectName(QString::fromUtf8("actionLoad_Example"));
        centralWidget = new QWidget(StereoDepthMap);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        centralWidget->setEnabled(true);
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(1);
        sizePolicy1.setVerticalStretch(1);
        sizePolicy1.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy1);
        centralWidget->setSizeIncrement(QSize(1, 1));
        centralWidget->setAcceptDrops(true);
        horizontalLayout_11 = new QHBoxLayout(centralWidget);
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setEnabled(true);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        tab->setEnabled(true);
        horizontalLayout_12 = new QHBoxLayout(tab);
        horizontalLayout_12->setSpacing(6);
        horizontalLayout_12->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        verticalLayout_10 = new QVBoxLayout();
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(10, -1, 10, -1);
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        verticalLayout_8 = new QVBoxLayout();
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        labelInput1 = new QLabel(tab);
        labelInput1->setObjectName(QString::fromUtf8("labelInput1"));

        horizontalLayout_7->addWidget(labelInput1);

        buttonLoad1 = new QPushButton(tab);
        buttonLoad1->setObjectName(QString::fromUtf8("buttonLoad1"));
        buttonLoad1->setEnabled(true);

        horizontalLayout_7->addWidget(buttonLoad1);


        verticalLayout_8->addLayout(horizontalLayout_7);

        viewImage1 = new QGraphicsView(tab);
        viewImage1->setObjectName(QString::fromUtf8("viewImage1"));

        verticalLayout_8->addWidget(viewImage1);


        horizontalLayout_9->addLayout(verticalLayout_8);

        verticalLayout_9 = new QVBoxLayout();
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        labelInput2 = new QLabel(tab);
        labelInput2->setObjectName(QString::fromUtf8("labelInput2"));

        horizontalLayout_8->addWidget(labelInput2);

        buttonLoad2 = new QPushButton(tab);
        buttonLoad2->setObjectName(QString::fromUtf8("buttonLoad2"));

        horizontalLayout_8->addWidget(buttonLoad2);


        verticalLayout_9->addLayout(horizontalLayout_8);

        viewImage2 = new QGraphicsView(tab);
        viewImage2->setObjectName(QString::fromUtf8("viewImage2"));

        verticalLayout_9->addWidget(viewImage2);


        horizontalLayout_9->addLayout(verticalLayout_9);

        formLayout = new QFormLayout();
        formLayout->setSpacing(6);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        labelDepth = new QLabel(tab);
        labelDepth->setObjectName(QString::fromUtf8("labelDepth"));

        formLayout->setWidget(1, QFormLayout::LabelRole, labelDepth);

        spinBoxDepth = new QSpinBox(tab);
        spinBoxDepth->setObjectName(QString::fromUtf8("spinBoxDepth"));
        spinBoxDepth->setAcceptDrops(true);
        spinBoxDepth->setAccelerated(true);
        spinBoxDepth->setMinimum(1);
        spinBoxDepth->setMaximum(255);
        spinBoxDepth->setValue(16);

        formLayout->setWidget(1, QFormLayout::FieldRole, spinBoxDepth);

        labelDataTerm = new QLabel(tab);
        labelDataTerm->setObjectName(QString::fromUtf8("labelDataTerm"));

        formLayout->setWidget(2, QFormLayout::LabelRole, labelDataTerm);

        labelAlpha = new QLabel(tab);
        labelAlpha->setObjectName(QString::fromUtf8("labelAlpha"));

        formLayout->setWidget(3, QFormLayout::LabelRole, labelAlpha);

        spinBoxAlpha = new QDoubleSpinBox(tab);
        spinBoxAlpha->setObjectName(QString::fromUtf8("spinBoxAlpha"));
        spinBoxAlpha->setSingleStep(0.1);
        spinBoxAlpha->setValue(5);

        formLayout->setWidget(3, QFormLayout::FieldRole, spinBoxAlpha);

        spinBoxDataTerm = new QSpinBox(tab);
        spinBoxDataTerm->setObjectName(QString::fromUtf8("spinBoxDataTerm"));
        spinBoxDataTerm->setMaximum(200);
        spinBoxDataTerm->setValue(50);

        formLayout->setWidget(2, QFormLayout::FieldRole, spinBoxDataTerm);

        labelTheta = new QLabel(tab);
        labelTheta->setObjectName(QString::fromUtf8("labelTheta"));

        formLayout->setWidget(4, QFormLayout::LabelRole, labelTheta);

        spinBoxTheta = new QDoubleSpinBox(tab);
        spinBoxTheta->setObjectName(QString::fromUtf8("spinBoxTheta"));
        spinBoxTheta->setMaximum(1);
        spinBoxTheta->setValue(0.5);

        formLayout->setWidget(4, QFormLayout::FieldRole, spinBoxTheta);

        verticalSpacer_11 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(0, QFormLayout::LabelRole, verticalSpacer_11);

        verticalSpacer_12 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(5, QFormLayout::LabelRole, verticalSpacer_12);


        horizontalLayout_9->addLayout(formLayout);

        horizontalLayout_9->setStretch(0, 3);
        horizontalLayout_9->setStretch(1, 3);
        horizontalLayout_9->setStretch(2, 2);

        verticalLayout_10->addLayout(horizontalLayout_9);

        verticalSpacer_4 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_10->addItem(verticalSpacer_4);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalSpacer_10 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_10);

        labelRegularizer = new QLabel(tab);
        labelRegularizer->setObjectName(QString::fromUtf8("labelRegularizer"));

        verticalLayout_4->addWidget(labelRegularizer);

        verticalSpacer_9 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_9);

        buttonTV = new QPushButton(tab);
        buttonTV->setObjectName(QString::fromUtf8("buttonTV"));

        verticalLayout_4->addWidget(buttonTV);

        buttonQuadratic = new QPushButton(tab);
        buttonQuadratic->setObjectName(QString::fromUtf8("buttonQuadratic"));

        verticalLayout_4->addWidget(buttonQuadratic);

        buttonHuber = new QPushButton(tab);
        buttonHuber->setObjectName(QString::fromUtf8("buttonHuber"));

        verticalLayout_4->addWidget(buttonHuber);

        verticalSpacer_8 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_8);

        pushSaveDepth = new QPushButton(tab);
        pushSaveDepth->setObjectName(QString::fromUtf8("pushSaveDepth"));

        verticalLayout_4->addWidget(pushSaveDepth);


        horizontalLayout_10->addLayout(verticalLayout_4);

        horizontalSpacer_4 = new QSpacerItem(50, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_4);

        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        buttonStop = new QPushButton(tab);
        buttonStop->setObjectName(QString::fromUtf8("buttonStop"));

        gridLayout->addWidget(buttonStop, 0, 3, 1, 1);

        labelMap = new QLabel(tab);
        labelMap->setObjectName(QString::fromUtf8("labelMap"));

        gridLayout->addWidget(labelMap, 0, 0, 1, 1);

        progressBar = new QProgressBar(tab);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(24);

        gridLayout->addWidget(progressBar, 0, 2, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 1, 1, 1);


        verticalLayout_7->addLayout(gridLayout);

        viewStereo1 = new QGraphicsView(tab);
        viewStereo1->setObjectName(QString::fromUtf8("viewStereo1"));

        verticalLayout_7->addWidget(viewStereo1);


        horizontalLayout_10->addLayout(verticalLayout_7);

        horizontalLayout_10->setStretch(0, 1);
        horizontalLayout_10->setStretch(2, 2);

        verticalLayout_10->addLayout(horizontalLayout_10);

        buttonExit1 = new QPushButton(tab);
        buttonExit1->setObjectName(QString::fromUtf8("buttonExit1"));

        verticalLayout_10->addWidget(buttonExit1);


        horizontalLayout_12->addLayout(verticalLayout_10);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        horizontalLayout_13 = new QHBoxLayout(tab_2);
        horizontalLayout_13->setSpacing(6);
        horizontalLayout_13->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(10, -1, 10, -1);
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setSizeConstraint(QLayout::SetMaximumSize);
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        label = new QLabel(tab_2);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout_5->addWidget(label);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalSpacer_3 = new QSpacerItem(10, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_3);

        viewStereo2 = new QGraphicsView(tab_2);
        viewStereo2->setObjectName(QString::fromUtf8("viewStereo2"));

        horizontalLayout_5->addWidget(viewStereo2);


        verticalLayout_5->addLayout(horizontalLayout_5);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_9 = new QLabel(tab_2);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout_4->addWidget(label_9);

        spinBoxDiscrete = new QSpinBox(tab_2);
        spinBoxDiscrete->setObjectName(QString::fromUtf8("spinBoxDiscrete"));

        horizontalLayout_4->addWidget(spinBoxDiscrete);


        verticalLayout_5->addLayout(horizontalLayout_4);

        label_6 = new QLabel(tab_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        verticalLayout_5->addWidget(label_6);

        verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_6);

        label_5 = new QLabel(tab_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        verticalLayout_5->addWidget(label_5);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_3);


        horizontalLayout_6->addLayout(verticalLayout_5);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label_2 = new QLabel(tab_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_3->addWidget(label_2);

        viewFocus = new QGraphicsView(tab_2);
        viewFocus->setObjectName(QString::fromUtf8("viewFocus"));
        viewFocus->setMouseTracking(true);

        verticalLayout_3->addWidget(viewFocus);


        horizontalLayout_6->addLayout(verticalLayout_3);

        horizontalLayout_6->setStretch(0, 1);
        horizontalLayout_6->setStretch(1, 2);

        verticalLayout->addLayout(horizontalLayout_6);

        verticalSpacer_2 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout->addItem(verticalSpacer_2);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_5);

        pushSaveResult = new QPushButton(tab_2);
        pushSaveResult->setObjectName(QString::fromUtf8("pushSaveResult"));

        verticalLayout_6->addWidget(pushSaveResult);

        pushReset = new QPushButton(tab_2);
        pushReset->setObjectName(QString::fromUtf8("pushReset"));

        verticalLayout_6->addWidget(pushReset);

        buttonExit2 = new QPushButton(tab_2);
        buttonExit2->setObjectName(QString::fromUtf8("buttonExit2"));

        verticalLayout_6->addWidget(buttonExit2);


        horizontalLayout_2->addLayout(verticalLayout_6);

        horizontalSpacer_2 = new QSpacerItem(50, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);

        tabWidget_2 = new QTabWidget(tab_2);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        formLayoutWidget_2 = new QWidget(tab_3);
        formLayoutWidget_2->setObjectName(QString::fromUtf8("formLayoutWidget_2"));
        formLayoutWidget_2->setGeometry(QRect(0, 10, 521, 141));
        formLayout_3 = new QFormLayout(formLayoutWidget_2);
        formLayout_3->setSpacing(6);
        formLayout_3->setContentsMargins(11, 11, 11, 11);
        formLayout_3->setObjectName(QString::fromUtf8("formLayout_3"));
        formLayout_3->setHorizontalSpacing(20);
        formLayout_3->setContentsMargins(0, 0, 0, 0);
        radioLinear = new QRadioButton(formLayoutWidget_2);
        radioLinear->setObjectName(QString::fromUtf8("radioLinear"));
        radioLinear->setChecked(false);

        formLayout_3->setWidget(0, QFormLayout::LabelRole, radioLinear);

        radioQuadratic = new QRadioButton(formLayoutWidget_2);
        radioQuadratic->setObjectName(QString::fromUtf8("radioQuadratic"));
        radioQuadratic->setChecked(true);
        radioQuadratic->setAutoRepeat(true);

        formLayout_3->setWidget(0, QFormLayout::FieldRole, radioQuadratic);

        spinBoxPowerBlur = new QSpinBox(formLayoutWidget_2);
        spinBoxPowerBlur->setObjectName(QString::fromUtf8("spinBoxPowerBlur"));
        spinBoxPowerBlur->setMaximum(1000);
        spinBoxPowerBlur->setValue(200);

        formLayout_3->setWidget(2, QFormLayout::LabelRole, spinBoxPowerBlur);

        label_3 = new QLabel(formLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout_3->setWidget(2, QFormLayout::FieldRole, label_3);

        pushBlurAgain = new QPushButton(formLayoutWidget_2);
        pushBlurAgain->setObjectName(QString::fromUtf8("pushBlurAgain"));

        formLayout_3->setWidget(4, QFormLayout::FieldRole, pushBlurAgain);

        verticalSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout_3->setItem(1, QFormLayout::LabelRole, verticalSpacer);

        verticalSpacer_7 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout_3->setItem(3, QFormLayout::FieldRole, verticalSpacer_7);

        tabWidget_2->addTab(tab_3, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QString::fromUtf8("tab_5"));
        spinBoxInpainting = new QSpinBox(tab_5);
        spinBoxInpainting->setObjectName(QString::fromUtf8("spinBoxInpainting"));
        spinBoxInpainting->setGeometry(QRect(120, 20, 65, 24));
        spinBoxInpainting->setMaximum(1000);
        spinBoxInpainting->setValue(200);
        label_4 = new QLabel(tab_5);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(190, 20, 141, 21));
        label_4->setScaledContents(false);
        label_4->setMargin(2);
        label_7 = new QLabel(tab_5);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(20, 90, 471, 16));
        label_8 = new QLabel(tab_5);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(20, 120, 411, 21));
        tabWidget_2->addTab(tab_5, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        formLayoutWidget_4 = new QWidget(tab_4);
        formLayoutWidget_4->setObjectName(QString::fromUtf8("formLayoutWidget_4"));
        formLayoutWidget_4->setGeometry(QRect(90, 10, 321, 141));
        formLayout_5 = new QFormLayout(formLayoutWidget_4);
        formLayout_5->setSpacing(6);
        formLayout_5->setContentsMargins(11, 11, 11, 11);
        formLayout_5->setObjectName(QString::fromUtf8("formLayout_5"));
        formLayout_5->setContentsMargins(0, 0, 0, 0);
        doubleSpinBox = new QDoubleSpinBox(formLayoutWidget_4);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setMaximum(20);
        doubleSpinBox->setValue(2);

        formLayout_5->setWidget(0, QFormLayout::LabelRole, doubleSpinBox);

        label_10 = new QLabel(formLayoutWidget_4);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        formLayout_5->setWidget(0, QFormLayout::FieldRole, label_10);

        tabWidget_2->addTab(tab_4, QString());

        horizontalLayout_2->addWidget(tabWidget_2);

        horizontalLayout_2->setStretch(0, 2);
        horizontalLayout_2->setStretch(1, 2);
        horizontalLayout_2->setStretch(2, 5);

        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));

        verticalLayout->addLayout(horizontalLayout_3);

        verticalLayout->setStretch(0, 2);
        verticalLayout->setStretch(1, 1);
        verticalLayout->setStretch(2, 1);

        horizontalLayout_13->addLayout(verticalLayout);

        tabWidget->addTab(tab_2, QString());

        horizontalLayout_11->addWidget(tabWidget);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(-1);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        horizontalLayout->setContentsMargins(10, -1, 10, -1);

        horizontalLayout_11->addLayout(horizontalLayout);

        StereoDepthMap->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(StereoDepthMap);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 866, 22));
        menuBar->setDefaultUp(true);
        menuLoad_Images = new QMenu(menuBar);
        menuLoad_Images->setObjectName(QString::fromUtf8("menuLoad_Images"));
        menuSave = new QMenu(menuBar);
        menuSave->setObjectName(QString::fromUtf8("menuSave"));
        menuExit = new QMenu(menuBar);
        menuExit->setObjectName(QString::fromUtf8("menuExit"));
        StereoDepthMap->setMenuBar(menuBar);
        mainToolBar = new QToolBar(StereoDepthMap);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        StereoDepthMap->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(StereoDepthMap);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        StereoDepthMap->setStatusBar(statusBar);

        menuBar->addAction(menuLoad_Images->menuAction());
        menuBar->addAction(menuSave->menuAction());
        menuBar->addAction(menuExit->menuAction());
        menuLoad_Images->addAction(actionLoadIm1);
        menuLoad_Images->addAction(actionLoadIm2);
        menuLoad_Images->addSeparator();
        menuLoad_Images->addAction(actionLoadDepth);
        menuLoad_Images->addSeparator();
        menuLoad_Images->addAction(actionLoad_Example);
        menuSave->addAction(actionSaveDepth);
        menuSave->addAction(actionSaveFocus);

        retranslateUi(StereoDepthMap);

        tabWidget->setCurrentIndex(1);
        tabWidget_2->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(StereoDepthMap);
    } // setupUi

    void retranslateUi(QMainWindow *StereoDepthMap)
    {
        StereoDepthMap->setWindowTitle(QApplication::translate("StereoDepthMap", "StereoDepthMap", 0, QApplication::UnicodeUTF8));
        actionLoadIm1->setText(QApplication::translate("StereoDepthMap", "Load Image 1", 0, QApplication::UnicodeUTF8));
        actionLoadIm2->setText(QApplication::translate("StereoDepthMap", "Load Image 2", 0, QApplication::UnicodeUTF8));
        actionSaveDepth->setText(QApplication::translate("StereoDepthMap", "Depth Map", 0, QApplication::UnicodeUTF8));
        actionSaveFocus->setText(QApplication::translate("StereoDepthMap", "Focused Image", 0, QApplication::UnicodeUTF8));
        actionLoadDepth->setText(QApplication::translate("StereoDepthMap", "Load Depth Map", 0, QApplication::UnicodeUTF8));
        actionLoad_Example->setText(QApplication::translate("StereoDepthMap", "Load Example", 0, QApplication::UnicodeUTF8));
        labelInput1->setText(QApplication::translate("StereoDepthMap", "Input Image 1", 0, QApplication::UnicodeUTF8));
        buttonLoad1->setText(QApplication::translate("StereoDepthMap", "Load", 0, QApplication::UnicodeUTF8));
        labelInput2->setText(QApplication::translate("StereoDepthMap", "Input Image 2", 0, QApplication::UnicodeUTF8));
        buttonLoad2->setText(QApplication::translate("StereoDepthMap", "Load", 0, QApplication::UnicodeUTF8));
        labelDepth->setText(QApplication::translate("StereoDepthMap", "Depth", 0, QApplication::UnicodeUTF8));
        labelDataTerm->setText(QApplication::translate("StereoDepthMap", "Data Term", 0, QApplication::UnicodeUTF8));
        labelAlpha->setText(QApplication::translate("StereoDepthMap", "Alpha", 0, QApplication::UnicodeUTF8));
        labelTheta->setText(QApplication::translate("StereoDepthMap", "Theta", 0, QApplication::UnicodeUTF8));
        labelRegularizer->setText(QApplication::translate("StereoDepthMap", "Choose Regularizer", 0, QApplication::UnicodeUTF8));
        buttonTV->setText(QApplication::translate("StereoDepthMap", "Total Variation", 0, QApplication::UnicodeUTF8));
        buttonQuadratic->setText(QApplication::translate("StereoDepthMap", "Quadratic", 0, QApplication::UnicodeUTF8));
        buttonHuber->setText(QApplication::translate("StereoDepthMap", "Huber", 0, QApplication::UnicodeUTF8));
        pushSaveDepth->setText(QApplication::translate("StereoDepthMap", "Save Depth Map", 0, QApplication::UnicodeUTF8));
        buttonStop->setText(QApplication::translate("StereoDepthMap", "Stop Calculation", 0, QApplication::UnicodeUTF8));
        labelMap->setText(QApplication::translate("StereoDepthMap", "Stereo Depth Map", 0, QApplication::UnicodeUTF8));
        buttonExit1->setText(QApplication::translate("StereoDepthMap", "Exit", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("StereoDepthMap", "Depth Map", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("StereoDepthMap", "Depth Map", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("StereoDepthMap", "Reduce levels in Depth Map to", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("StereoDepthMap", "0 means no reducing.", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("StereoDepthMap", "Clicking in Result will choose a level.", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("StereoDepthMap", "Result Image", 0, QApplication::UnicodeUTF8));
        pushSaveResult->setText(QApplication::translate("StereoDepthMap", "Save Result", 0, QApplication::UnicodeUTF8));
        pushReset->setText(QApplication::translate("StereoDepthMap", "Reset", 0, QApplication::UnicodeUTF8));
        buttonExit2->setText(QApplication::translate("StereoDepthMap", "Exit", 0, QApplication::UnicodeUTF8));
        radioLinear->setText(QApplication::translate("StereoDepthMap", "Linear Blur", 0, QApplication::UnicodeUTF8));
        radioQuadratic->setText(QApplication::translate("StereoDepthMap", "Quadratic Blur", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("StereoDepthMap", "Power of Blur", 0, QApplication::UnicodeUTF8));
        pushBlurAgain->setText(QApplication::translate("StereoDepthMap", "Blur current again", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_3), QApplication::translate("StereoDepthMap", "Focus", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("StereoDepthMap", "Number of Iterations", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("StereoDepthMap", "Clicking again will add the inpainting to the current picture.", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("StereoDepthMap", "Click \"Reset\" to start fresh.", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_5), QApplication::translate("StereoDepthMap", "Inpainting", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("StereoDepthMap", "Intensity of Highlight", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_4), QApplication::translate("StereoDepthMap", "Highlighting", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("StereoDepthMap", "Image Focus ", 0, QApplication::UnicodeUTF8));
        menuLoad_Images->setTitle(QApplication::translate("StereoDepthMap", "Load Images", 0, QApplication::UnicodeUTF8));
        menuSave->setTitle(QApplication::translate("StereoDepthMap", "Save", 0, QApplication::UnicodeUTF8));
        menuExit->setTitle(QApplication::translate("StereoDepthMap", "Exit", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class StereoDepthMap: public Ui_StereoDepthMap {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STEREODEPTHMAP_H
