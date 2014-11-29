#ifndef LOOGER_H
#define LOGGER_H


#include <iostream>
#include <vector>
#include <string>
#include <stack>

#include "aux.h"    //  new ostream types

using namespace std;


class Logger
{
    public:

        Logger(const int prio = 0);
        Logger(const string& msg, const int prio = 0);
        ~Logger();

        void push(const string& parent);
        void pop(bool success = true);
        void eol();

        template <class T>
        Logger& operator<< (const T& info)
        {
            if (maxLevel_ < 0 || getCurrentLevel() <= maxLevel_)
            {
                if(!interrupts_.empty()) 
                {
                    if (!interrupts_.top()) {
                        cout << endl;
                        interrupts_.top() = true;
                        eol_ = true;
                    }
                }

                if (eol_) {
                    indent();
                    cout << "|-- ";
                    eol_ = false;
                }

                cout << info;
            }

            return *this;
        };

    private:

        int prio_;


    public: 

        static void setConfirmationType(const string& type) { confType_ = type; };
        static void setMaxLevel(const int level);

        static const string& getConfirmation();
        static const int getCurrentLevel();


    private: 

        static bool eol_;
        static int maxLevel_;
        static string confType_;

        static stack<string> parents_;
        static stack<Timer> timers_;
        static stack<bool> interrupts_;

        static vector<string> confirmations_;
        static vector<string> confirmationsGirl_;
        static vector<string> confirmationsDoge_;

        static void indent();
        static void printTime(Timer& timer);

};


//  formatted output
std::ostream& operator<< (std::ostream& out, const dim3& d);
std::ostream& operator<< (std::ostream& out, const float2& v);


#endif