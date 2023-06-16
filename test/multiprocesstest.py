#problem: how to reference multiple subprocesses neatly without hardcoding?

def holderguy(*args):
    while True:
        for x in args[0]:
            print(x.values())

if __name__ == "__main__":
    import multiprocessing 
    multiprocessing.freeze_support()
    shared_mem_manager = multiprocessing.Manager()
    shared_pool_meta_dict = shared_mem_manager.dict()
    analyze_pool_count = 4
    for x in range(analyze_pool_count):
        #init analyzed/keycount dicts
        #init raw dicts
        shared_pool_meta_dict[x] = shared_mem_manager.dict()
        #start the subprocesses
        #give kivy the list of subprocesses 
    