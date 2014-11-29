#include "stereodepthmap.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    StereoDepthMap w;
    //w.initializeEventFilter();
    w.show();

    return a.exec();
}
