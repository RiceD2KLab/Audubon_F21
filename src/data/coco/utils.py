import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """
    Provides helper functions for tracking a series of values and providing access to smoothed values over a window or the global series average.
    Will be used to track key metrics during training and testing for deep learning models.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Creates an instance of a series of values using a "deque" (double ended queue).
        Deques are a data structure which allow elements to be added/removed on both ends. 

        Input:
            window_size (int): The size of the moving window used to calculate the average. Default is 20.
            fmt (str): The string format used to print the output. Default is "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Updates the deque with a chosen value, and updates the count of elements.
        
        Input:
            value: The new value to add to the deque.
            n: Specifies how many times to add the value. Default is 1. 
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        Computes the median of values in the deque.
        
        Output:
            The median value of the deque.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        Computes the mean of values in the sliding window.
        
        Output:
            The mean value of the sliwing window.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        Computes the global average of all values that have been added to the deque instance.
        
        Output:
            The global average of all values that have been added to the sliding window.
        """
        return self.total / self.count

    @property
    def max(self):
        """
        Returns the maximum value in the deque.
        
        Output:
            The maximum value in the deque.
        """
        return max(self.deque)

    @property
    def value(self):
        """
        Returns the most recent value added to the deque.
        
        Output:
            The most recent value added to the deque.
        """
        return self.deque[-1]

    def __str__(self):
        """
        Returns key statistics of the deque.
        
        Output:
            A formatted string containing the deque median, sliding window average, global average, maximum  value
            and most recent value.
        """
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    
    Input:
        data: any picklable object
    
    Output:
        list[data]: list of data gathered from each process.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    
    Input:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    
    Output:
        A dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    """
    This class provides a way to track metrics during training and testing for a deep learning model.
    """
    def __init__(self, delimiter="\t"):
        """
        Creates an instance of a metric that will be tracked during training and testing. 
        
        Input:
            Delimiter (str): Separator to use between metric values and names when displaying them.  
        """
        # Create a dictionary (meter) that maps metric names to SmoothedValue instances
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the values of the meters in the logger.

        Input:
            **kwargs: A dictionary containing the key-value pairs to update. The keys represent the name of 
            the meters to update and the values represent the corresponding values to add to the meter.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        This method is called when an attribute that is not defined in the class is accessed.
        Returns the requested attribute by checking in three places:
            1. If the requested attribute is the key of a meter in the meters dictionary, it returns the meter.
            2. If the requested attribute is already in the object's dictionary, it returns the attribute from there.
            3. Otherwise, it raises an AttributeError with an informative message.
            
        Input:
            attr (str): The name of the attribute that is being requested.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """
        Returns a string representation of the metrics stored in the `MetricLogger` object.
        
        Output:
            A string containing the name and value of each metric, separated by the delimiter specified during initialization.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronize the state of all the meters across all processes in the distributed environment.
        This method calls the `synchronize_between_processes` method of each meter instance in the `meters` dictionary.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Adds a new meter to the metric logger.

        Input:
            name (str): Name of the meter.
            meter (SmoothedValue): The meter to be added.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log and print the status of an iterable, such as training or validation progress, at a given frequency.

        Input:
            iterable: An iterable to log and print the status of.
            print_freq (int): The frequency at which to log and print the status of the iterable.
            header (str, optional): The header string to use when logging and printing the status of the iterable.

        Output:
            A series of printed statements tracking the status of training or validation progress.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    """
    Collates a batch of samples into a tuple of tensors.
    
    Input:
        batch: A list of tuples, where each tuple contains the data for one sample.
        
    Output:
        A tuple of tensors, where each tensor contains the data for all the samples in the batch.
    """
    return tuple(zip(*batch))


def mkdir(path):
    """
    Creates a new directory at a given path. Raises an error if there is a error besides the inputted directory already
    existing. 
    
    Input:
        path (str): The path at which to create the directory.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This method sets up the printing behavior for distributed training. If the current process is not the master process, 
    it disables printing.
    
    Input:
        is_master (bool): whether the current process is the master process or not.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    This function checks whether distributed training is available and has been initialized.

    Output:
        True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns the number of processes in the distributed group, including the master process. 
    
    Output:
        The number of process in the distributed group, including the master process. If the distributed environment is 
        not available or not initialized, it returns 1, which corresponds to the case where no distributed training is 
        being used.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns the rank of the current process in the distributed group. 
    
    Output:
       The rank of the current process in the distributed group, or 0 if not part of a distributed group.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Returns a boolean indicating whether the current process is the main process.
    
    Output:
        A boolean value. True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save the given object(s) using torch.save() if the current process is the main process.

    Input:
        *args: positional arguments to pass to torch.save().
        **kwargs: keyword arguments to pass to torch.save().
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initializes distributed mode for PyTorch training based on the environment variables.
    
    Intput:
        args: Namespace containing the command-line arguments.
    """
    # Check if the environment variables "RANK", "WORLD_SIZE", and "LOCAL_RANK" are set.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        #If they are set, set the rank, world size, and GPU device ID for the process.
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # If not, check if the environment variable "SLURM_PROCID" is set and use it to set the rank and GPU device ID.
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    # If neither set of environment variables is found, disable distributed mode.
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    # Once the rank and GPU device ID are determined, set the device for PyTorch and initializes the process group using the NCCL 
    # backend and the URL specified in the command-line arguments.
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
