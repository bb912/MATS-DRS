import torch.multiprocessing as multiprocessing


def child(queue: multiprocessing.SimpleQueue, i):
    print(i)
    queue.put(i + 10)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", True)
    sim_round_queue = multiprocessing.Queue()
    procs = []
    for i in range(4):
        p = multiprocessing.Process(target=child, args=(sim_round_queue, i))
        p.start()
        procs.append(p)

    while True:
        a = sim_round_queue.get()
        print(a)
