import engines.engine as engine
import threading

thread0 = threading.Thread(target=engine.process_camera, args=(0, 0))
thread1 = threading.Thread(target=engine.process_camera, args=(1, 1))

thread0.start()
thread1.start()

thread0.join()
thread1.join()