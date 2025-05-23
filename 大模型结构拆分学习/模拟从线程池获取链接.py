import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

# ========== 1. 定义连接池类 ==========
class ConnectionPool:
    def __init__(self, max_connections=3):
        self.max_connections = max_connections
        self._lock = threading.Lock()  # 线程锁
        self._connections = queue.Queue(maxsize=max_connections)  # 连接队列
        
        # 初始化固定数量的连接
        for _ in range(max_connections):
            self._connections.put(self._create_connection())

    def _create_connection(self):
        """模拟创建连接（例如数据库连接）"""
        return f"Connection-{time.time()}"  # 返回唯一标识符代替真实连接

    def get_connection(self, timeout=5):
        """从池中获取一个连接（阻塞直到获取或超时）"""
        try:
            return self._connections.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("获取连接超时")

    def release_connection(self, conn):
        """将连接放回池中"""
        with self._lock:
            if self._connections.full():
                raise Exception("连接池已满，无法归还")
            self._connections.put(conn)

    def close_all(self):
        """关闭所有连接（清理资源）"""
        while not self._connections.empty():
            conn = self._connections.get()
            print(f"关闭连接: {conn}")

# ========== 2. 定义线程任务 ==========
def worker_task(pool, task_id):
    """线程任务：获取连接 -> 执行操作 -> 释放连接"""
    try:
        # 获取连接
        conn = pool.get_connection()
        print(f"任务 {task_id} 获取连接: {conn}")
        
        # 模拟使用连接执行操作
        time.sleep(1)  # 假设这里执行SQL查询等操作
        print(f"任务 {task_id} 完成操作")
        
    except Exception as e:
        print(f"任务 {task_id} 出错: {e}")
    finally:
        # 确保释放连接
        '''
        locals() 函数详解‌
        ‌基本特性‌
            ‌功能‌：返回当前作用域中所有‌局部变量‌构成的字典，键为变量名，值为对应的对象引用13。
        ‌作用域‌：
            在函数内调用时，仅包含该函数的局部变量；
            在模块层级调用时，返回的字典与 globals() 的输出类似（包含全局变量
        '''
        if 'conn' in locals():
            pool.release_connection(conn)
            print(f"任务 {task_id} 释放连接: {conn}")

# ========== 3. 主程序逻辑 ==========
if __name__ == "__main__":
    # 创建连接池（最多3个连接）
    pool = ConnectionPool(max_connections=3)

    # 创建线程池（4个线程，超过连接池容量以演示等待）
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交8个任务
        futures = [executor.submit(worker_task, pool, i) for i in range(8)]
        
        # 等待所有任务完成（可选）
        for future in futures:
            future.result()

    # 关闭所有连接
    pool.close_all()
