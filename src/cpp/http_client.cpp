#include <cstring>
#include <iostream>
#include <emscripten/fetch.h>

class HttpClient
{
public:
    static void get(const char *url);
    static void post(const char *url, const char *data);

private:
    static void downloadSucceeded(emscripten_fetch_t *fetch);
    static void downloadFailed(emscripten_fetch_t *fetch);
};

void HttpClient::downloadSucceeded(emscripten_fetch_t *fetch)
{
    std::cout << "下载成功: " << fetch->numBytes << " bytes\n";
    std::cout << "数据: " << fetch->data << "\n";
    emscripten_fetch_close(fetch);
}

void HttpClient::downloadFailed(emscripten_fetch_t *fetch)
{
    std::cout << "下载失败! HTTP 状态码: " << fetch->status << "\n";
    emscripten_fetch_close(fetch);
}

void HttpClient::get(const char *url)
{
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);

    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = downloadSucceeded;
    attr.onerror = downloadFailed;

    const char *headers[] = {
        "Content-Type", "application/json",
        "Accept", "application/json",
        nullptr};
    attr.requestHeaders = headers;

    emscripten_fetch(&attr, url);
}

void HttpClient::post(const char *url, const char *data)
{
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);

    strcpy(attr.requestMethod, "POST");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = downloadSucceeded;
    attr.onerror = downloadFailed;

    const char *headers[] = {
        "Content-Type", "application/json",
        nullptr};
    attr.requestHeaders = headers;

    attr.requestData = data;
    attr.requestDataSize = strlen(data);

    emscripten_fetch(&attr, url);
}

// 添加导出函数
extern "C" {
    void make_http_get(const char* url) {
        HttpClient::get(url);
    }

    void make_http_post(const char* url, const char* data) {
        HttpClient::post(url, data);
    }
}
