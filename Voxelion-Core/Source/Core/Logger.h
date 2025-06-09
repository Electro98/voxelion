#pragma once


class Log {
public:
    enum LogLevel {
        DebugLevel = 5,
        TraceLevel = 4,
        InfoLevel = 3,
        WarningLevel = 2,
        ErrorLevel = 1,
        CriticalLevel = 0,
    };

public:
    static void Initialize();
    static void SetLogLevel(LogLevel);

    static void Debug(const char[]); 
    static void Trace(const char[]); 
    static void Info(const char[]); 
    static void Warning(const char[]); 
    static void Error(const char[]); 
    static void WTF(const char[]); 
private:
    Log();
    static Log Instance;

private:
    LogLevel m_CurrentLogLevel;

    // void logMessage(const LogLevel level, const char msg[]);
    template <Log::LogLevel level>
    void logMessage(const char msg[]);
};
