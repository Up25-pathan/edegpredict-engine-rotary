/**
 * @file AsyncExporter.h
 * @brief Asynchronous file export with background thread
 * 
 * This class provides non-blocking file I/O by using a worker thread
 * that processes export jobs from a queue. The simulation loop can
 * continue without waiting for disk writes to complete.
 */

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <memory>
#include <iostream>

namespace edgepredict {

/**
 * @brief Export job that can be queued for async processing
 */
struct ExportJob {
    std::function<void()> task;
    int step;
    double time;
};

/**
 * @brief Asynchronous exporter with background worker thread
 * 
 * Usage:
 *   AsyncExporter async;
 *   async.start();
 *   async.enqueue([&]() { vtk.exportStep(step, time, mesh); });
 *   // ... simulation continues immediately ...
 *   async.stop(); // Wait for all jobs to complete
 */
class AsyncExporter {
public:
    AsyncExporter() : m_running(false), m_jobsCompleted(0), m_jobsQueued(0) {}
    
    ~AsyncExporter() {
        stop();
    }
    
    /**
     * @brief Start the background worker thread
     */
    void start() {
        if (m_running) return;
        
        m_running = true;
        m_workerThread = std::thread(&AsyncExporter::workerLoop, this);
        std::cout << "[AsyncExporter] Started background I/O thread" << std::endl;
    }
    
    /**
     * @brief Stop the worker thread and wait for pending jobs
     */
    void stop() {
        if (!m_running) return;
        
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_running = false;
        }
        m_condition.notify_one();
        
        if (m_workerThread.joinable()) {
            m_workerThread.join();
        }
        
        std::cout << "[AsyncExporter] Stopped. Processed " << m_jobsCompleted 
                  << " export jobs." << std::endl;
    }
    
    /**
     * @brief Queue an export job for async processing
     * @param job The export task to execute
     * 
     * This method returns immediately. The job will be processed
     * by the background thread when it becomes available.
     */
    void enqueue(ExportJob job) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.push(std::move(job));
            m_jobsQueued++;
        }
        m_condition.notify_one();
    }
    
    /**
     * @brief Convenience method to enqueue a simple task
     */
    void enqueue(std::function<void()> task, int step = 0, double time = 0.0) {
        enqueue(ExportJob{std::move(task), step, time});
    }
    
    /**
     * @brief Get number of pending jobs in queue
     */
    size_t pendingJobs() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }
    
    /**
     * @brief Get total jobs completed
     */
    int completedJobs() const {
        return m_jobsCompleted;
    }
    
    /**
     * @brief Wait for all pending jobs to complete
     */
    void flush() {
        while (pendingJobs() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
private:
    void workerLoop() {
        while (true) {
            ExportJob job;
            
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this] {
                    return !m_queue.empty() || !m_running;
                });
                
                if (!m_running && m_queue.empty()) {
                    break;
                }
                
                if (!m_queue.empty()) {
                    job = std::move(m_queue.front());
                    m_queue.pop();
                }
            }
            
            // Execute the export job outside the lock
            if (job.task) {
                try {
                    job.task();
                    m_jobsCompleted++;
                } catch (const std::exception& e) {
                    std::cerr << "[AsyncExporter] Error in job (step " << job.step 
                              << "): " << e.what() << std::endl;
                }
            }
        }
    }
    
    std::thread m_workerThread;
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    std::queue<ExportJob> m_queue;
    
    std::atomic<bool> m_running;
    std::atomic<int> m_jobsCompleted;
    std::atomic<int> m_jobsQueued;
};

} // namespace edgepredict
