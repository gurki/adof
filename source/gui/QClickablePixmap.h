#ifndef Q_CLICKABLE_PIXMAP_H
#define Q_CLICKABLE_PIXMAP_H


#include <QtGui/QGraphicsPixmapItem>
#include <QtGui/QMouseEvent>
#include <QtGui/QGraphicsSceneMouseEvent>

#include "stereodepthmap.h"


class QClickablePixmap : public QGraphicsPixmapItem
{
    public:

        QClickablePixmap(QGraphicsItem* parent = 0) :
            QGraphicsPixmapItem(parent),
            handler_(NULL)
        {};

        QClickablePixmap (const QPixmap& pixmap, QGraphicsItem* parent = 0) : 
            QGraphicsPixmapItem(pixmap, parent),
            handler_(NULL)
        {};

        QClickablePixmap (const QImage& image, StereoDepthMap* handler = NULL) : 
            QGraphicsPixmapItem(QPixmap::fromImage(image))
        {
            setMousePressEventHandler(handler);
        };

        void setMousePressEventHandler(StereoDepthMap* handler) { 
            handler_ = handler; 
        };

        // void setPixmap (const QPixmap& pixmap) { QGraphicsPixmapItem::setPixmap(pixmap); };


    protected:

        virtual void mousePressEvent (QGraphicsSceneMouseEvent* event) 
        {
            if (handler_) {
                handler_->didPressButton(event->pos().x(), event->pos().y());
            }
        };

        virtual void mouseMoveEvent (QGraphicsSceneMouseEvent* event)
        {
            if (handler_) {
                handler_->didMoveMouse(event->pos().x(), event->pos().y());
            }
        };


    private:

        StereoDepthMap* handler_;
};

#endif