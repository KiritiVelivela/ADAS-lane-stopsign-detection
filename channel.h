#include <queue>
#include <mutex>
#include <exception>
#include <condition_variable>
#include <optional>

class ChannelClosedException : public std::exception
{
public:
    ChannelClosedException() : message_("Channel is closed") {}

    const char *what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

template <typename T>
class Channel
{
public:
    Channel() : open_(true) {}

    void send(const T &data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (data_)
        {
            cv_send_.wait(lock);
        }
        data_ = data;
        cv_receive_.notify_one();
    }

    void send(const T &&data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (data_)
        {
            cv_send_.wait(lock);
        }
        data_ = std::move(data);
        cv_receive_.notify_one();
    }

    T receive()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!data_ && open_)
        {
            cv_receive_.wait(lock);
        }
        if (!open_ && !data_)
        {
            throw ChannelClosedException();
        }
        T receivedData = std::move(data_.value());
        data_ = std::nullopt;
        cv_send_.notify_one();
        return std::move(receivedData);
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = false;
        cv_receive_.notify_all();
    }

private:
    std::optional<T> data_;
    bool open_;
    std::mutex mutex_;
    std::condition_variable cv_send_;
    std::condition_variable cv_receive_;
};

template <typename T>
class BufferedChannel
{
public:
    explicit BufferedChannel(size_t capacity) : capacity_(capacity) {}

    void send(const T &data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_)
        {
            cv_send_.wait(lock);
        }
        queue_.push(data);
        cv_receive_.notify_one();
    }

    void send(const T &&data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_)
        {
            cv_send_.wait(lock);
        }
        queue_.push(std::move(data));
        cv_receive_.notify_one();
    }

    T receive()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty() && !closed_)
        {
            cv_receive_.wait(lock);
        }
        if (queue_.empty() && closed_)
        {
            throw ChannelClosedException();
        }
        T data = std::move(queue_.front());
        queue_.pop();
        cv_send_.notify_one();
        return std::move(data);
    }

    void close()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        cv_receive_.notify_all();
    }

    bool isClosed() const { return closed_; }

private:
    std::queue<T> queue_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_send_;
    std::condition_variable cv_receive_;
    bool closed_ = false;
};