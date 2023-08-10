import numpy as np
import jax
# from functools import reduce
import logging


class Aggregation:
    def __call__(self, *xs):
        self.n = len(xs)
        assert self.n > 0, "xs should not be empty"

        x = xs[0]

        if x is None:
            return None
        elif isinstance(x, list) and isinstance(x[0], (np.ndarray, np.generic, jax.Array)):
            return self.aggregate_list_of_array(*xs)
        elif isinstance(x, int) or isinstance(x, float):
            return self.aggregate_scalars(*xs)
        elif isinstance(x, (np.ndarray, np.generic, jax.Array)):
            return self.aggregate_arrays(*xs)
        else:
            logging.error(f"weight type not correct, got {type(x)}")
            logging.error("expect int, float, np.ndarray, list of np.ndarray")
            raise NotImplementedError(f"{type(x)} is not supported")



class SumUp(Aggregation):

    def aggregate_list_of_array(self, *xs):
        agg_weight = []
        len_of_weight = len(xs[0])
        for j in range(len_of_weight):

            tmp_array = xs[0][j]
            for i in range(1, self.n):
                tmp_array = tmp_array + xs[i][j]

            agg_weight.append(tmp_array)

        return agg_weight

    def aggregate_scalars(self, *xs):
        return sum(xs)

    def aggregate_arrays(self, *xs):
        result = xs[0]
        for i in range(1, self.n):
            result = result + xs[i]

        return result



# class Intersect(Aggregation):

#     def aggregate_list_of_array(self, xs):
#         raise NotImplementedError("Intersect for list of array is not implemented yet")

#     def aggregate_scalars(self, xs):
#         raise NotImplementedError("Intersect for scalars is not implemented yet")

#     def aggregate_arrays(self, xs):
#         if len(xs[0].shape) > 1:
#             raise NotImplementedError("Intersect for is only implemented for 1D array")

#         # index
#         intersect_list = reduce(np.intersect1d, xs)

#         # reorder
#         ordered_list = []
#         for i in xs[0]:
#             if i in intersect_list:
#                 ordered_list.append(i)

#         return ordered_list



class CheckSame(Aggregation):

    def aggregate_list_of_array(self, xs):
        raise NotImplementedError("CheckSame for list of array is not implemented yet")

    def aggregate_scalars(self, xs):
        if len(set(xs)) > 1:
            logging.error(f"weight from client are not the same {xs}")
            raise RuntimeError

        return xs[0]

    def aggregate_arrays(self, xs):
        if not all_equal(xs):
            logging.error(f"weight from client are not the same {xs}")
            raise RuntimeError

        return xs[0]


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


# class Union(Aggregation):

#     def aggregate_list_of_array(self, xs):
#         raise NotImplementedError("Union for list of array is not implemented yet")

#     def aggregate_scalars(self, xs):
#         return list(set(xs))

#     def aggregate_arrays(self, xs):
#         if len(xs[0].shape) > 1:
#             raise NotImplementedError("Union for is only implemented for 1D array")

#         # index
#         union_list = reduce(np.union1d, xs)

#         # reorder
#         ordered_list = []
#         for i in xs[0]:
#             if i in union_list:
#                 ordered_list.append(i)

#         return ordered_list
