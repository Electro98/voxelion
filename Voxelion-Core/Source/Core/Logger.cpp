#include "Logger.h"

#include <format>
#include <iostream>
#include <string>

Log Log::Instance;

void Log::Initialize() {
    Log::Instance = {};
}

void Log::SetLogLevel(LogLevel level) {
    Log::Instance.m_CurrentLogLevel = level;
}

void Log::Debug(const char message[]) {
    Log::Instance.logMessage<DebugLevel>(message);
}
void Log::Trace(const char message[]) {
    Log::Instance.logMessage<TraceLevel>(message);
}
void Log::Info(const char message[]) {
    Log::Instance.logMessage<InfoLevel>(message);
}
void Log::Warning(const char message[]) {
    Log::Instance.logMessage<WarningLevel>(message);
}
void Log::Error(const char message[]) {
    Log::Instance.logMessage<ErrorLevel>(message);
}
void Log::WTF(const char message[]) {
    Log::Instance.logMessage<CriticalLevel>(message);
}

Log::Log() {
    m_CurrentLogLevel = DebugLevel;
}

template <class T> struct dependent_false : std::false_type {};

constexpr const char *levelToStr(Log::LogLevel level) {
    switch (level) {
        case Log::DebugLevel:
            return "Debug";
        case Log::TraceLevel:
            return "Trace";
        case Log::InfoLevel:
            return "Info";
        case Log::WarningLevel:
            return "Warning";
        case Log::ErrorLevel:
            return "Error";
        case Log::CriticalLevel:
            return "WTF?!";
        default:
            return (const char*) 0;
    }
} 

template<Log::LogLevel level> void Log::logMessage(const char msg[]) {
    if (level > m_CurrentLogLevel) {
        return;
    }
    std::string message = std::format("[{}]: {}", levelToStr(level), msg);

    std::cout << message << std::endl;
}

