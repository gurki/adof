/****************************************************************************
** Meta object code from reading C++ file 'stereodepthmap.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "stereodepthmap.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'stereodepthmap.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_StereoDepthMap[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      15,   32,   32,   32, 0x08,
      33,   32,   32,   32, 0x08,
      43,   32,   32,   32, 0x08,
      56,   32,   32,   32, 0x08,
      69,   32,   32,   32, 0x08,
      82,   32,   32,   32, 0x08,
      97,   32,   32,   32, 0x08,
     111,   32,   32,   32, 0x08,
     126,   32,   32,   32, 0x08,
     139,   32,   32,   32, 0x08,
     157,   32,   32,   32, 0x08,
     175,   32,   32,   32, 0x08,
     183,   32,   32,   32, 0x08,
     194,   32,   32,   32, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_StereoDepthMap[] = {
    "StereoDepthMap\0startQuadratic()\0\0"
    "startTV()\0startHuber()\0openImage1()\0"
    "openImage2()\0openDepthMap()\0openExample()\0"
    "saveDepthMap()\0saveResult()\0"
    "updateFocusBlur()\0loadDiscreteGPU()\0"
    "reset()\0setAbort()\0exitThis()\0"
};

void StereoDepthMap::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        StereoDepthMap *_t = static_cast<StereoDepthMap *>(_o);
        switch (_id) {
        case 0: _t->startQuadratic(); break;
        case 1: _t->startTV(); break;
        case 2: _t->startHuber(); break;
        case 3: _t->openImage1(); break;
        case 4: _t->openImage2(); break;
        case 5: _t->openDepthMap(); break;
        case 6: _t->openExample(); break;
        case 7: _t->saveDepthMap(); break;
        case 8: _t->saveResult(); break;
        case 9: _t->updateFocusBlur(); break;
        case 10: _t->loadDiscreteGPU(); break;
        case 11: _t->reset(); break;
        case 12: _t->setAbort(); break;
        case 13: _t->exitThis(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData StereoDepthMap::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject StereoDepthMap::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_StereoDepthMap,
      qt_meta_data_StereoDepthMap, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &StereoDepthMap::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *StereoDepthMap::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *StereoDepthMap::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_StereoDepthMap))
        return static_cast<void*>(const_cast< StereoDepthMap*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int StereoDepthMap::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
