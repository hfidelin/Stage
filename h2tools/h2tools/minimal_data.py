from __future__ import print_function, absolute_import, division

__all__ = ['MinimalData']

class MetaData(type):
    """
    Metaclass to check if all necessary methods are presented.
    """
    def __new__(mcs, name, bases, dicts):
        required = ['check_far', 'compute_aux', 'divide', '__len__']
        missing = []
        for method in required:
            if not method in dicts:
                missing.append(method)
        if missing:
            raise NotImplementedError("Methods {} are not implemented in "
                "class {}. See docstrings of corresponding methods of class "
                "``h2tools.MinimalData`` for help".format(missing, name))
        return super(MetaData, mcs).__new__(mcs, name, bases, dicts)

class MinimalData(object):
    """
    Minimal set of methods for any data object to work with ``h2tools``.

    If it is used as base class for class of data objects, it checks if all
    necessary functions are presented in the time of initialization of data
    object.
    """
    def __new__(cls, *args, **kwargs):
        if cls.check_far.__func__ is MinimalData.check_far.__func__:
            raise NotImplementedError("Method ``check_far`` of {} is not "
                "implemented. See docstring of ``MinimalData.check_far`` for "
                "more information.".format(cls))
        if cls.compute_aux.__func__ is MinimalData.compute_aux.__func__:
            raise NotImplementedError("Method ``compute_aux`` of {} is not "
                "implemented. See docstring of ``MinimalData.compute_aux`` for"
                " more information.".format(cls))
        if cls.divide.__func__ is MinimalData.divide.__func__:
            raise NotImplementedError("Method ``divide`` of {} is not "
                "implemented. See docstring of ``MinimalData.divide`` for"
                " more information.".format(cls))
        if cls.__len__.__func__ is MinimalData.__len__.__func__:
            raise NotImplementedError("Method ``divide`` of {} is not "
                "implemented. See docstring of ``MinimalData.divide`` for"
                " more information.".format(cls))

    def check_far(self, self_aux, other_aux):
        """
        Checks if clusters are far from each other by auxiliary data.

        Auxiliary data can be anything, i.e. bounding box. This function
        must be symmetric (transitive to parameters ``self_aux`` and
        ``other_aux``).

        Parameters
        ----------
        self_aux, other_aux : Python objects
            Auxiliary data for two clusters.

        Returns
        -------
        boolean
            ``True`` if clusters are far, ``False`` otherwise.
        """
        raise NotImplementedError("Method ``check_far`` of {} is not "
                "implemented. See docstring of ``MinimalData.check_far`` for "
                "more information.".format(self.__class__))

    def compute_aux(self, index):
        """
        Computes auxiliary data for cluster, corresponding to ``index``.

        Simplest example of such an auxiliary data is bounding box.

        Parameters
        ----------
        index : 1-dimensional array
            Indices of objects in initial cluster, corresponding to
            given subcluster.

        Returns
        -------
        Python object
            Some auxiliary data.
        """
        raise NotImplementedError("Method ``compute_aux`` of {} is not "
                "implemented. See docstring of ``MinimalData.compute_aux`` for"
                " more information.".format(self.__class__))

    def divide(self, index):
        """
        Divides cluster, corresponding to ``index``.

        Parameters
        ----------
        index : 1-dimensional array
            Indices of objects in initial cluster, corresponding to
            given subcluster.

        Returns
        -------
        permutation : 1-dimensional array
            How to permute indices of given cluster, such that indices
            of new subclusters are successive.
        division : list
            Indices of ``i``-th subcluster take places from ``division[i]``
            inclusively to ``division[i+1]`` exclusively.

        Notes
        -----
        Length of resulting ``division`` equals number of sublcusters
        plus 1. Simple example: ``division = [0, 1, 3, 5, 10]`` means
        cluster was divided into 4 subclusters, first subclusters has
        only 1 item, second subcluster has 2 items, third subcluster
        has 2 items and last subcluster has 5 items.
        """
        raise NotImplementedError("Method ``divide`` of {} is not "
                "implemented. See docstring of ``MinimalData.divide`` for"
                " more information.".format(self.__class__))

    def __len__(self):
        """
        Returns number of objects or items in cluster.
        """
        raise NotImplementedError("Method ``__len__`` of {} is not "
                "implemented. See docstring of ``MinimalData.__len__`` for"
                " more information.".format(self.__class__))
