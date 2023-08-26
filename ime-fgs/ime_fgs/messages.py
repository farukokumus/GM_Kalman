# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

from __future__ import annotations

import numpy as np
import math

from ime_fgs.unscented_utils import unscented_transform_gaussian
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian, \
    moment_matched_weighted_mean_info_of_doubly_truncated_gaussian
from ime_fgs.utils import inherit_method_docs, col_vec, mat, try_inv_else_robinv

from abc import ABC, abstractmethod
from enum import Enum


class PortMessageDirection(Enum):
    Undefined = 0
    Forward = 1
    Backward = 2


class Message(ABC):
    """
    Abstract base class for all messages being passed on a factor graph.

    Provides raising `NotImplementedError` for all methods a message type could implement.
    """

    def __init__(self, direction):
        assert isinstance(direction, PortMessageDirection)
        self.direction = direction

    def convert(self, target_type) -> Message:
        """
        Convert this message to another message type.

        :param target_type: The message type that this message should be converted to.
        :return: The converted message.
        """
        raise NotImplementedError('Convert method not implemented for this datatype.')

    @abstractmethod
    def combine(self, msg_b, auto_convert=False, try_other=True) -> Message:
        """
        Combine the information from this message with information from another message.

        This corresponds to the multiplication of two probability distributions both representing information on the
        same random variable.

        :param msg_b: The second message providing information on the underlying variable.
        :param auto_convert: Automatically convert one or both of the involved messages to enable combination. May be
          very inefficient.
        :param try_other: If self.combine(other_msg) is not implemented, try other_msg.combine(self, try_other=False).
        :return: A message containing the combined information of the two provided messages.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        else:
            return not result

    def multiply_deterministic(self, matrix, inverse=False):
        """
        Multiply the underlying random variable by a deterministic factor (matrix).

        :param matrix: The matrix by which the variable is multiplied.
        :param inverse: If `True`, the inverse direction is calculated.
        :return: The modified message.
        """
        raise NotImplementedError('Deterministic multiplication method not implemented for this type.')

    @staticmethod
    @abstractmethod
    def non_informative(n):
        """
        Return non-informative message of dimensionality n.

        :param n: Number of dimensions of the message mean.
        :return: Non-informative message of dimensionality n.
        """
        pass

    @abstractmethod
    def is_non_informative(self):
        """
        Return whether this message is non-informative.

        :return: True or False.
        """
        pass


class MultipleCombineMessage(ABC):
    @staticmethod
    @abstractmethod
    def combine_multiple(msg_list):
        pass


@inherit_method_docs
class GaussianMeanCovMessage(Message):
    def __init__(self, mean, cov, direction=PortMessageDirection.Undefined):
        """
        Create a Gaussian message parameterized by mean and covariance
        :param mean: array_like, a nx1 mean vector
        :param cov: array_like, a nxn covariance matrix
        :param direction:
        """
        super().__init__(direction)
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        # assert np.allclose(self.cov, self.cov.T)
        assert np.ndim(self.mean) == np.ndim(self.cov) == 2
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]
        assert self.mean.shape[1] == 1

    def combine(self, msg_b, auto_convert=False, try_other=True):
        super().combine(msg_b)

        if try_other:
            try:
                return msg_b.combine(self, auto_convert=False, try_other=False)
            except NotImplementedError:
                pass

        # use auto convert only as last resort
        if auto_convert:
            try:
                return self.convert(GaussianWeightedMeanInfoMessage).combine(msg_b, auto_convert=True)
            except NotImplementedError:
                pass

            if try_other:
                try:
                    return msg_b.combine(self, auto_convert=True, try_other=False)
                except NotImplementedError:
                    pass

        raise NotImplementedError('Combine is unimplemented for this data type (inefficient)')

    def convert(self, target_type):
        if target_type is type(self):
            return self
        elif target_type is GaussianWeightedMeanInfoMessage:
            if self.is_degenerate():
                raise NotImplementedError
            info = np.linalg.inv(self.cov)
            weighted_mean = info @ self.mean
            return GaussianWeightedMeanInfoMessage(weighted_mean, info)

        elif target_type is GaussianMixtureMeanCovMessage:
            if self.is_non_informative():
                return GaussianMixtureMeanCovMessage.non_informative(1, self.cov.shape[1])
            else:
                return GaussianMixtureMeanCovMessage([[1]], [self.mean], [self.cov])

        elif target_type is GaussianMixtureWeightedMeanInfoMessage:
            if self.is_non_informative():
                return GaussianMixtureWeightedMeanInfoMessage.non_informative(1, self.cov.shape[1])
            else:
                info = np.linalg.inv(self.cov)
                weighted_mean = info @ self.mean
                return GaussianMixtureWeightedMeanInfoMessage([[1]], [weighted_mean], [info])
        else:
            raise NotImplementedError('This kind of message type conversion has not been implemented yet.')

    def multiply_deterministic(self, matrix, inverse=False):
        if inverse:
            raise NotImplementedError('Backward multiplication is unimplemented for this data type (inefficient)')
        else:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()
            mean = matrix @ self.mean
            cov = matrix @ self.cov @ matrix_h

            return GaussianMeanCovMessage(mean, cov)

    @staticmethod
    def non_informative(n, direction=PortMessageDirection.Undefined, inf_approx=None):
        mean = np.zeros((n, 1))
        cov = np.zeros((n, n))
        for i in range(0, n):
            if inf_approx is not None:
                cov[i, i] = inf_approx
            else:
                cov[i, i] = np.inf
        return GaussianMeanCovMessage(mean, cov, direction=direction)

    def is_non_informative(self):
        return all(self.cov == self.non_informative(self.cov.shape[0]).cov)

    def is_degenerate(self):
        _, eigs, _ = np.linalg.svd(self.cov)
        return eigs[-1] == 0

    def unscented_transform(self, func, sigma_point_scheme=None, alpha=None):
        mean, cov, cr_var = unscented_transform_gaussian(self.mean, self.cov, func,
                                                         sigma_point_scheme=sigma_point_scheme, alpha=alpha)

        return GaussianMeanCovMessage(mean, cov), cr_var

    def approximate_truncation_by_moment_matching(self, hyperplane_normal, upper_bounds, lower_bounds,
                                                  inverse=False):
        # if inverse:
        # todo: Check whether the inverse direction is really no different.
        # else:

        moment_matched_mean, moment_matched_cov = \
            moment_matched_mean_cov_of_doubly_truncated_gaussian(self.mean, self.cov, hyperplane_normal,
                                                                 upper_bounds, lower_bounds)

        return GaussianMeanCovMessage(moment_matched_mean, moment_matched_cov)

    @staticmethod
    def get_means(list_of_GMC_msgs):
        assert all([isinstance(msg, GaussianMeanCovMessage) for msg in list_of_GMC_msgs])
        means = np.array( np.concatenate([msg.mean for msg in list_of_GMC_msgs], axis=1))
        assert isinstance(means, np.ndarray)
        N = list_of_GMC_msgs[0].mean.shape[0]
        assert means.shape == (N, len(list_of_GMC_msgs))
        return means

    @staticmethod
    def get_covs(list_of_GMC_msgs):
        assert all([isinstance(msg, GaussianMeanCovMessage) for msg in list_of_GMC_msgs])
        covs = np.dstack([msg.cov for msg in list_of_GMC_msgs])
        assert isinstance(covs, np.ndarray)
        N = list_of_GMC_msgs[0].mean.shape[0]
        assert covs.shape == (N, N, len(list_of_GMC_msgs))
        return covs

    def __add__(self, other):
        if isinstance(other, self.__class__):
            mean = self.mean + other.mean
            cov = self.cov + other.cov
            return GaussianMeanCovMessage(mean, cov)
        else:
            return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return GaussianMeanCovMessage(-self.mean, self.cov)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.allclose(self.mean, other.mean) and np.allclose(self.cov, other.cov)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            info_dividend = np.linalg.inv(self.cov)
            info_divisor = np.linalg.inv(other.cov)
            cov = np.linalg.inv(info_dividend - info_divisor)
            # if cov < 0:
            #     raise NotImplementedError( '''Negative covariances do not make sense.''' )
            mean = cov @ (info_dividend @ self.mean - info_divisor @ other.mean)
            return GaussianMeanCovMessage(mean, cov)
        else:
            return NotImplemented

    def __repr__(self):
        return "GaussianMeanCovMessage(" + repr(self.mean.tolist()) + ", " + repr(self.cov.tolist()) + ")"

    def __str__(self):
        return "Mean:\n" + str(self.mean) + ",\nCovariance Matrix:\n" + str(self.cov)


@inherit_method_docs
class GaussianWeightedMeanInfoMessage(Message, MultipleCombineMessage):
    def __init__(self, weighted_mean, info, direction=PortMessageDirection.Undefined):
        """
        Create a Gaussian message parameterized by the weighted mean and the information matrix
        :param weighted_mean: array_like, a nx1 weighted mean vector
        :param info: array_like, a nxn information matrix
        :param direction:
        """
        super().__init__(direction)
        self.weighted_mean = np.array(weighted_mean)
        self.info = np.array(info)
        self.info = (self.info + self.info.T) / 2
        # assert np.allclose(self.info, self.info.T)
        assert np.ndim(self.weighted_mean) == 2 == np.ndim(self.info)
        assert self.weighted_mean.shape[0] == self.info.shape[0] == self.info.shape[1]
        assert self.weighted_mean.shape[1] == 1

    def combine(self, msg_b, auto_convert=False, try_other=True):
        if isinstance(msg_b, GaussianWeightedMeanInfoMessage) or auto_convert:
            if not isinstance(msg_b, GaussianWeightedMeanInfoMessage):
                try:
                    msg_b = msg_b.convert(GaussianWeightedMeanInfoMessage)
                except NotImplementedError:
                    if try_other:
                        return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
                    else:
                        raise NotImplementedError

            assert self.weighted_mean.shape == msg_b.weighted_mean.shape

            weighted_mean = self.weighted_mean + msg_b.weighted_mean
            info = self.info + msg_b.info

            return GaussianWeightedMeanInfoMessage(weighted_mean, info)
        elif try_other:
            return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError

    @staticmethod
    def combine_multiple(msg_list):
        # TODO: Implement directionality checks?
        assert all([isinstance(msg, GaussianWeightedMeanInfoMessage) for msg in msg_list])

        weighted_mean = sum(msg.weighted_mean for msg in msg_list)
        info = sum(msg.info for msg in msg_list)

        return GaussianWeightedMeanInfoMessage(weighted_mean, info)

    def convert(self, target_type):
        if target_type is type(self):
            return self
        elif target_type is GaussianMeanCovMessage:
            # TODO: Find out, why non informative out_msgs are set in equality node
            if self.is_non_informative():
                return GaussianMeanCovMessage.non_informative(self.info.shape[0])
            else:
                cov = np.linalg.inv(self.info)
                mean = cov @ self.weighted_mean
                return GaussianMeanCovMessage(mean, cov)

        elif target_type is GaussianMixtureMeanCovMessage:
            if self.is_non_informative():
                return GaussianMixtureMeanCovMessage.non_informative(1, self.info.shape[1])
            else:
                cov = np.linalg.inv(self.info)
                mean = cov @ self.weighted_mean
                return GaussianMixtureMeanCovMessage([[1]], [mean], [cov])

        elif target_type is GaussianMixtureWeightedMeanInfoMessage:
            if self.is_non_informative():
                return GaussianMixtureWeightedMeanInfoMessage.non_informative(1, self.info.shape[1])
            else:
                return GaussianMixtureWeightedMeanInfoMessage([[1]], [self.weighted_mean], [self.info])

        elif target_type is GaussianTildeMessage:
            if self.is_non_informative():
                return GaussianTildeMessage.non_informative(self.info.shape[0])
            else:
                raise NotImplementedError('This kind of message type conversion has not been implemented yet.')
        else:
            raise NotImplementedError('This kind of message type conversion has not been implemented yet.')

    def multiply_deterministic(self, matrix, inverse=False):

        if inverse:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()
            weighted_mean = matrix_h @ self.weighted_mean
            info = matrix_h @ self.info @ matrix

            return GaussianWeightedMeanInfoMessage(weighted_mean, info)

        else:
            raise NotImplementedError('''Forward multiplication for this data type is unimplemented''')

    def is_non_informative(self):
        # check if all entries are zero
        return not self.info.any()

    @staticmethod
    def non_informative(n, direction=PortMessageDirection.Undefined, inf_approx=None):
        weighted_mean = np.zeros((n, 1))
        info = np.zeros((n, n))
        if inf_approx is not None:
            for i in range(0, n):
                info[i, i] = 1 / inf_approx
        return GaussianWeightedMeanInfoMessage(weighted_mean, info, direction=direction)

    def unscented_transform(self, func, sigma_point_scheme=None, alpha=None):
        raise NotImplementedError

    def approximate_truncation_by_moment_matching(self, hyperplane_normal, upper_bounds, lower_bounds, inverse=False):
        # if inverse:
        # todo: Check whether the inverse direction is really no different.
        # else:

        moment_matched_weighted_mean, moment_matched_info = \
            moment_matched_weighted_mean_info_of_doubly_truncated_gaussian(self.weighted_mean, self.info,
                                                                           hyperplane_normal,
                                                                           upper_bounds,
                                                                           lower_bounds)

        return GaussianWeightedMeanInfoMessage(moment_matched_weighted_mean, moment_matched_info)

    def __add__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return GaussianWeightedMeanInfoMessage(-self.weighted_mean, self.info)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.allclose(self.weighted_mean, other.weighted_mean) and np.allclose(self.info, other.info)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            weighted_mean = self.weighted_mean - other.weighted_mean
            info = self.info - other.info
            return GaussianWeightedMeanInfoMessage(weighted_mean, info)
        else:
            return NotImplemented

    def __repr__(self):
        return "GaussianWeightedMeanInfoMessage(" + repr(self.weighted_mean.tolist()) + ", " + \
               repr(self.info.tolist()) + ")"

    def __str__(self):
        return "Weighted Mean:\n" + str(self.weighted_mean) + ",\nInformation Matrix:\n" + str(self.info)


@inherit_method_docs
class GaussianTildeMessage(Message):
    """
    For implementation details, refer to:
    * Loeliger, Bruderer et al. (2016): On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing
    * Loeliger et al. (2007): The factor graph approach to model-based signal processing
    * Petersen, Hoffmann, Rostalski (2018): On Approximate Nonlinear Gaussian Message Passing on Factor Graphs
    """

    def __init__(self, xi, W, direction=PortMessageDirection.Undefined):
        super().__init__(direction)
        self.xi = col_vec(xi)
        self.W = mat(W)
        assert np.allclose(self.W, self.W.T)
        assert np.ndim(self.xi) == 2 == np.ndim(self.W)
        assert self.xi.shape[0] == self.W.shape[0] == self.W.shape[1]
        assert self.xi.shape[1] == 1

    def combine(self, msg_b, auto_convert=False, try_other=True):
        if isinstance(msg_b, GaussianMeanCovMessage):
            if self.direction == PortMessageDirection.Backward:
                # This (tilde) message is a backwards message, at least from the point of view of the node its exerted
                # from. Hence, we interpret the other message as a forward message, regardless of the directionality of
                # the other port.
                # --> This should enable things like connecting two in ports or two out ports.
                mean = msg_b.mean - msg_b.cov @ self.xi
            elif self.direction == PortMessageDirection.Forward:
                mean = msg_b.mean + msg_b.cov @ self.xi
            else:
                # Don't know anything about the directionality of this message, hence can't determine how to interpret
                # the other message.
                raise RuntimeError('Cannot calculate marginal from tilde message since port directionality unknown.')
            cov = msg_b.cov - msg_b.cov @ self.W @ msg_b.cov
            return GaussianMeanCovMessage(mean, cov, PortMessageDirection.Undefined)
        elif try_other:
            return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError('Combine is not implemented for this data type.')

    def convert(self, target_type, other_msg=None):
        if target_type is type(self):
            return self
        elif other_msg is None:
            raise NotImplementedError('Cannot convert tilde message without information about the other message.')
        elif isinstance(other_msg, GaussianMeanCovMessage):
            n = self.W.shape[0]
            if self == self.non_informative(n):
                # At least one of the fwd and bwd msgs is noninformative
                if other_msg == GaussianNonInformativeMessage(n):
                    raise RuntimeError('Cannot convert since uninformative.')
                else:
                    return target_type.non_informative(n, direction=self.direction)
            else:
                W_inv = try_inv_else_robinv(self.W)

                cov = W_inv - other_msg.cov
                if self.direction == PortMessageDirection.Forward:
                    mean = W_inv @ self.xi + other_msg.mean
                elif self.direction == PortMessageDirection.Backward:
                    mean = W_inv @ self.xi - other_msg.mean
                else:
                    raise RuntimeError('Cannot convert tilde message of direction is unknown.')
                return GaussianMeanCovMessage(mean, cov, direction=self.direction).convert(target_type)
        else:
            raise NotImplementedError('No message conversion implemented for this other_msg type.')

    def multiply_deterministic(self, matrix, inverse=False):
        if inverse:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()
            xi = matrix_h @ self.xi
            W = matrix_h @ self.W @ matrix
            return GaussianTildeMessage(xi, W)
        else:
            raise NotImplementedError('Forward multiplication is not implemented for this data type.')

    def is_non_informative(self):
        return all(self.W == self.non_informative(self.W.shape[0]).W)

    @staticmethod
    def non_informative(n, direction=PortMessageDirection.Undefined, inf_approx=None):
        xi = np.zeros(n)
        W = np.zeros((n, n))
        if inf_approx is not None:
            for i in range(0, n):
                W[i, i] = 1 / inf_approx
        return GaussianTildeMessage(xi, W, direction=direction)

    def __add__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __radd__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __sub__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __rsub__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.allclose(self.xi, other.xi) and np.allclose(self.W, other.W)
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def __repr__(self):
        return "GaussianTildeMessage(" + repr(self.xi.tolist()) + ", " + repr(self.W.tolist()) + ")"

    def __str__(self):
        return "Xi:\n" + str(self.xi) + ",\nW Tilde Matrix:\n" + str(self.W)


@inherit_method_docs
class GaussianNonInformativeMessage(GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage,
                                    GaussianTildeMessage):

    def __init__(self, N, direction=PortMessageDirection.Undefined, inf_approx=None):
        Message.__init__(self, direction)
        self.N = N
        nonInformativeMeanCov = GaussianMeanCovMessage.non_informative(N, inf_approx=inf_approx)
        self.mean = nonInformativeMeanCov.mean
        self.cov = nonInformativeMeanCov.cov
        nonInformativeWeightedMeanInfo = GaussianWeightedMeanInfoMessage.non_informative(N, inf_approx=inf_approx)
        self.weighted_mean = nonInformativeWeightedMeanInfo.weighted_mean
        self.info = nonInformativeWeightedMeanInfo.info
        nonInformativeTilde = GaussianTildeMessage.non_informative(N, inf_approx=inf_approx)
        self.xi = nonInformativeTilde.xi
        self.W = nonInformativeTilde.W

    def combine(self, msg_b, auto_convert=False, try_other=True):
        if try_other:
            return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError('Combine is not implemented for this data type.')

    def convert(self, target_type):
        return target_type.non_informative(self.N)

    def multiply_deterministic(self, matrix, inverse=False):
        return self

    def is_non_informative(self):
        return True

    def __add__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented

    def __eq__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def __repr__(self):
        return "GaussianNonInformativeMessage(N=" + str(self.N) + ")"

    def __str__(self):
        return "N:\n" + str(self.N)


@inherit_method_docs
class GaussianMixtureMeanCovMessage(Message):
    weights: np.ndarray
    mean: np.ndarray
    cov: np.ndarray

    def __init__(self, weights: list[list[float]], mean: list[list[list[float]]], cov: list[list[list[float]]], direction=PortMessageDirection.Undefined):
        """
        Create a Gaussian message parameterized by mean and covariance
        :param weights: array_like, a mx1 weight vector
        :param mean: array_like, a mxnx1 mean maxtrix
        :param cov: array_like, a mxnxn covariance tensor
        :param direction:
        """
        super().__init__(direction)
        self.weights = np.array(weights)
        self.mean = np.array(mean)
        self.cov = np.array(cov)

        assert math.isclose(np.sum(self.weights), 1.0)
        assert np.ndim(self.weights) == 2
        assert np.ndim(self.mean) == 3
        assert np.ndim(self.cov) == 3

        assert self.mean.shape[0] == self.weights.shape[0]
        assert self.mean.shape[1] == self.cov.shape[1] == self.cov.shape[2]
        assert self.mean.shape[2] == 1
        assert self.weights.shape[1] == 1

    def combine(self, msg_b: Message, auto_convert:bool=False, try_other:bool=True) -> Message:
        super().combine(msg_b)

        if try_other:
            try:
                return msg_b.combine(self, auto_convert=False, try_other=False)
            except NotImplementedError:
                pass

        # use auto convert only as last resort
        if auto_convert:
            try:
                return self.convert(GaussianMixtureWeightedMeanInfoMessage).combine(msg_b, auto_convert=True)
            except NotImplementedError:
                pass

            if try_other:
                try:
                    return msg_b.combine(self, auto_convert=True, try_other=False)
                except NotImplementedError:
                    pass

        raise NotImplementedError(f'Combine is unimplemented for {self.__class__.__name__} (inefficient)')

    def convert(self, target_type: type) -> Message:
        if target_type is type(self):
            return self

        elif target_type is GaussianMeanCovMessage:
            if self.is_non_informative():
                return GaussianMeanCovMessage.non_informative(self.info.shape[1])
            elif self.weights.shape[0] == 1:
                return GaussianMeanCovMessage(self.mean[0,:,:], self.cov[0,:,:])

        elif target_type is GaussianWeightedMeanInfoMessage:
            if self.is_non_informative():
                return GaussianWeightedMeanInfoMessage.non_informative(self.info.shape[1])
            elif self.weights.shape[0] == 1:
                info = np.linalg.inv(self.cov[0,:,:])
                weighted_mean = info @ self.mean[0,:,:]
                return GaussianWeightedMeanInfoMessage(weighted_mean, info)

        elif target_type is GaussianMixtureWeightedMeanInfoMessage:
            if self.is_degenerate():
                raise NotImplementedError
            info = np.zeros(self.cov.shape)
            weighted_mean = np.zeros(self.mean.shape)
            for i in range(self.weights.shape[0]):
                info[i,:,:] = np.linalg.inv(self.cov[i,:,:])
                weighted_mean[i,:,:] = info[i,:,:] @ self.mean[i,:,:]
            return GaussianMixtureWeightedMeanInfoMessage(self.weights.copy(), weighted_mean, info)

        else:
            raise NotImplementedError('This kind of message type conversion has not been implemented yet.')

    def multiply_deterministic(self, matrix: np.ndarray, inverse=False) -> GaussianMixtureMeanCovMessage:
        if inverse:
            raise NotImplementedError(f'Backward multiplication is unimplemented for {self.__class__.__name__} (inefficient)')
        else:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()

            mean = []
            cov = []

            for i in range(self.weights.shape[0]):
                mean += [matrix @ self.mean[i,:,:]]
                cov += [matrix @ self.cov[i,:,:] @ matrix_h]

            return GaussianMixtureMeanCovMessage(self.weights.copy(), mean, cov)

    @staticmethod
    def non_informative(m:int, n:int, direction=PortMessageDirection.Undefined, inf_approx:float=np.inf) -> GaussianMixtureMeanCovMessage:
        weights = np.full((m, 1), 1/m)
        mean = np.zeros((m, n, 1))
        cov = np.zeros((m, n, n))
        for i in range(m):
            for k in range(n):
                cov[i, k, k] = inf_approx
        return GaussianMixtureMeanCovMessage(weights, mean, cov, direction=direction)

    def is_non_informative(self) -> bool:
        return np.allclose(self.cov, self.non_informative(*self.cov.shape[:2]).cov)

    def is_degenerate(self) -> bool:
        for i in range(self.weights.shape[0]):
            _, eigs, _ = np.linalg.svd(self.cov[i, :, :])
            if eigs[-1] == 0:
                return True
        return False

    def unscented_transform(self, func, sigma_point_scheme=None, alpha=None) -> tuple[GaussianMixtureMeanCovMessage, np.ndarray]:

        mean = np.zeros(self.mean.shape)
        cov = np.zeros(self.cov.shape)
        cr_var = np.zeros(self.cov.shape)

        for i in range(self.weights.shape[0]):
            mean[i,:,:], cov[i,:,:], cr_var[i,:,:] = unscented_transform_gaussian(self.mean[i,:,:], self.cov[i,:,:], func,
                                                                           sigma_point_scheme=sigma_point_scheme, alpha=alpha)

        return GaussianMixtureMeanCovMessage(self.weights.copy(), mean, cov), cr_var

    def approximate_truncation_by_moment_matching(self, hyperplane_normal: np.ndarray, upper_bounds: float, lower_bounds: float,
                                                  inverse=False) -> GaussianMixtureMeanCovMessage:
        # if inverse:
        # todo: Check whether the inverse direction is really no different.
        # else:

        mean = np.zeros(self.mean.shape)
        cov = np.zeros(self.cov.shape)

        moment_match_func = moment_matched_mean_cov_of_doubly_truncated_gaussian
        for i in range(self.weights.shape[0]):
            mean[i,:,:], cov[i,:,:] = moment_match_func(self.mean[i,:,:], self.cov[i,:,:], hyperplane_normal,
                                                                    upper_bounds, lower_bounds)

        return GaussianMixtureMeanCovMessage(self.weights.copy(), mean, cov)

    @staticmethod
    def get_means(message_list: list[GaussianMixtureMeanCovMessage]) -> np.ndarray:
        assert np.all([isinstance(msg, GaussianMixtureMeanCovMessage) for msg in message_list])
        means = np.array(np.innerconcatenate([msg.mean for msg in message_list], axis=2))
        assert isinstance(means, np.ndarray)
        m = message_list[0].mean.shape[0]
        n = message_list[0].mean.shape[1]
        assert means.shape == (m, n, len(message_list))
        return means

    @staticmethod
    def get_covs(message_list: list[GaussianMixtureMeanCovMessage]) -> np.ndarray:
        assert all([isinstance(msg, GaussianMixtureMeanCovMessage) for msg in message_list])
        covs = np.dstack([msg.cov for msg in message_list])
        assert isinstance(covs, np.ndarray)
        m = message_list[0].mean.shape[0]
        n = message_list[0].mean.shape[1]
        assert covs.shape == (m, n, n, len(message_list))
        return covs

    def __add__(self, other):
        if isinstance(other, self.__class__):
            assert self.mean.shape[1:] == other.mean.shape[1:]
            assert self.cov.shape[1:] == other.cov.shape[1:]

            n = self.mean.shape[1]
            num_gaussians_self = self.weights.shape[0]
            num_gaussians_other = other.weights.shape[0]
            num_gaussians = num_gaussians_self * num_gaussians_other

            weights = np.zeros((num_gaussians, 1))
            mean = np.zeros((num_gaussians, n, 1))
            cov = np.zeros((num_gaussians, n, n))
            l = 0
            for i in range(num_gaussians_self):
                for k in range(num_gaussians_other):
                    weights[l] = self.weights[i] * other.weights[k]
                    mean[l, :, :] = self.mean[i, :, :] + other.mean[k, :, :]
                    cov[l, :, :] = self.cov[i, :, :] + other.cov[k, :, :]
                    l += 1
            return GaussianMixtureMeanCovMessage(weights, mean, cov)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return GaussianMixtureMeanCovMessage(self.weights.copy(), -self.mean.copy(), self.cov.copy())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.allclose(self.weights, other.weights) and\
                np.allclose(self.mean, other.mean) and\
                np.allclose(self.cov, other.cov)
        raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            assert self.mean.shape[1:] == other.mean.shape[1:]
            assert self.cov.shape[1:] == other.cov.shape[1:]

            n = self.mean.shape[1]
            num_gaussians_self = self.weights.shape[0]
            num_gaussians_other = other.weights.shape[0]
            num_gaussians = num_gaussians_self * num_gaussians_other

            weights = np.zeros((num_gaussians, 1))
            mean = np.zeros((num_gaussians, n, 1))
            cov = np.zeros((num_gaussians, n, n))
            l = 0
            for i in range(num_gaussians_self):
                for k in range(num_gaussians_other):
                    weights[l] = self.weights[i] * other.weights[k]
                    info_dividend = np.linalg.inv(self.cov[i, :, :])
                    info_divisor = np.linalg.inv(other.cov[k, :, :])
                    cov[l, :, :] = np.linalg.inv(info_dividend - info_divisor)
                    mean[l, :, :] = cov[l, :, :] @ (info_dividend @ self.mean[i, :, :] - info_divisor @ other.mean[k, :, :])
                    l += 1

            return GaussianMixtureMeanCovMessage(weights, mean, cov)
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.weights.tolist())}, {repr(self.mean.tolist())}, {repr(self.cov.tolist())})"

    def __str__(self):
        return f"Weights:\n{str(self.weights)},\nMean:\n{str(self.mean)},\nCovariance Matrix:\n{str(self.cov)}"


@inherit_method_docs
class GaussianMixtureWeightedMeanInfoMessage(Message, MultipleCombineMessage):

    weights:np.ndarray
    weighted_mean:np.ndarray
    info:np.ndarray

    def __init__(self, weights:list[list[float]], weighted_mean:list[list[list[float]]], info:list[list[list[float]]], direction=PortMessageDirection.Undefined):
        """
        Create a Gaussian message parameterized by the weighted mean and the information matrix
        :param weighted_mean: array_like, a nx1 weighted mean vector
        :param info: array_like, a nxn information matrix
        :param direction:
        """
        super().__init__(direction)
        self.weights = np.array(weights)
        self.weighted_mean = np.array(weighted_mean)
        self.info = np.array(info)

        assert math.isclose(np.sum(self.weights), 1.0)
        assert np.ndim(self.weights) == 2
        assert np.ndim(self.weighted_mean) == 3
        assert np.ndim(self.info) == 3

        assert self.weighted_mean.shape[0] == self.weights.shape[0]
        assert self.weighted_mean.shape[1] == self.info.shape[1] == self.info.shape[2]
        assert self.weighted_mean.shape[2] == 1
        assert self.weights.shape[1] == 1

        for i in range(self.weights.shape[0]):
            self.info[i,:,:] = (self.info[i,:,:] + self.info[i,:,:].T) / 2

    def combine(self, other, auto_convert=False, try_other=True):
        if isinstance(other, self.__class__) or auto_convert:
            if not isinstance(other, self.__class__):
                try:
                    other = other.convert(self.__class__)
                except NotImplementedError:
                    if try_other:
                        return other.combine(self, auto_convert=auto_convert, try_other=False)
                    else:
                        raise NotImplementedError

            assert self.weighted_mean.shape[1:] == other.weighted_mean.shape[1:]
            assert self.info.shape[1:] == other.info.shape[1:]

            n = self.weighted_mean.shape[1]
            num_gaussians_self = self.weights.shape[0]
            num_gaussians_other = other.weights.shape[0]
            num_gaussians = num_gaussians_self * num_gaussians_other

            weights = np.zeros((num_gaussians, 1))
            weighted_mean = np.zeros((num_gaussians, n, 1))
            info = np.zeros((num_gaussians, n, n))
            l = 0
            for i in range(num_gaussians_self):
                for k in range(num_gaussians_other):
                    weights[l] = self.weights[i] * other.weights[k]
                    weighted_mean[l, :, :] = self.weighted_mean[i, :, :] + other.weighted_mean[k, :, :]
                    info[l, :, :] = self.info[i, :, :] + other.info[k, :, :]
                    l += 1

            return GaussianMixtureWeightedMeanInfoMessage(weights, weighted_mean, info)

        elif try_other:
            return other.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError

    @staticmethod
    def combine_multiple(msg_list:list[GaussianMixtureWeightedMeanInfoMessage]) -> GaussianMixtureWeightedMeanInfoMessage:
        # TODO: Implement directionality checks?
        assert np.all([isinstance(msg, GaussianMixtureWeightedMeanInfoMessage) for msg in msg_list])

        if len(msg_list) > 0:
            this_msg = msg_list[0]
        else:
            raise NotImplementedError

        for msg in msg_list[1:]:
            this_msg = this_msg.combine(msg)

        return this_msg

    def convert(self, target_type: type) -> Message:
        if target_type is type(self):
            return self
        elif target_type is GaussianWeightedMeanInfoMessage:
            if self.is_non_informative():
                return GaussianWeightedMeanInfoMessage.non_informative(self.info.shape[1])
            elif self.weights.shape[0] == 1:
                    return GaussianWeightedMeanInfoMessage(self.weighted_mean[0,:,:], self.info[0,:,:])

        elif target_type is GaussianMixtureMeanCovMessage:
            if self.is_non_informative():
                return GaussianMixtureMeanCovMessage.non_informative(*self.info.shape[:2])
            else:
                cov = np.zeros(self.info.shape)
                mean = np.zeros(self.weighted_mean.shape)
                for i in range(self.weights.shape[0]):
                    cov[i,:,:] = np.linalg.inv(self.info[i,:,:])
                    mean[i,:,:] = cov[i,:,:] @ self.weighted_mean[i,:,:]
                return GaussianMixtureMeanCovMessage(self.weights.copy(), mean, cov)

        elif target_type is GaussianMeanCovMessage:
            if self.is_non_informative():
                return GaussianMeanCovMessage.non_informative(self.info.shape[1])
            elif self.weights.shape[0] == 1:
                    cov = np.linalg.inv(self.info[0,:,:])
                    mean = cov @ self.weighted_mean[0,:,:]
                    return GaussianMeanCovMessage(mean, cov)

        elif target_type is GaussianTildeMessage:
            if self.is_non_informative():
                return GaussianTildeMessage.non_informative(self.info.shape[0])
            else:
                raise NotImplementedError('This kind of message type conversion has not been implemented yet.')
        else:
            raise NotImplementedError('This kind of message type conversion has not been implemented yet.')

    def multiply_deterministic(self, matrix: np.ndarray, inverse=False) -> GaussianMixtureWeightedMeanInfoMessage:

        if inverse:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()

            weighted_mean = []
            info = []

            for i in range(self.weights.shape[0]):
                weighted_mean += [matrix_h @ self.weighted_mean[i,:,:]]
                info += [matrix_h @ self.info[i,:,:] @ matrix]

            return GaussianMixtureWeightedMeanInfoMessage(self.weights.copy(), weighted_mean, info)

        else:
            raise NotImplementedError(f'Forward multiplication is unimplemented for {self.__class__.__name__} (inefficient)')

    def is_non_informative(self):
        # check if all entries are zero
        return not self.info.any()

    @staticmethod
    def non_informative(m:int, n:int, direction=PortMessageDirection.Undefined, inf_approx:float=np.inf) -> GaussianMixtureWeightedMeanInfoMessage:
        weights = np.full((m, 1), 1/m)
        weighted_mean = np.zeros((n, 1))
        info = np.zeros((n, n))
        for i in range(m):
            for k in range(n):
                info[i, k, k] = 1 / inf_approx
        return GaussianMixtureWeightedMeanInfoMessage(weights, weighted_mean, info, direction=direction)

    def unscented_transform(self, func, sigma_point_scheme=None, alpha=None):
        raise NotImplementedError

    def approximate_truncation_by_moment_matching(self, hyperplane_normal: np.ndarray,  upper_bounds: float, lower_bounds: float,
                                                  inverse=False) -> GaussianMixtureWeightedMeanInfoMessage:
        # if inverse:
        # todo: Check whether the inverse direction is really no different.
        # else:

        weighted_mean = np.zeros(self.weighted_mean.shape)
        info = np.zeros(self.info.shape)
        moment_match_func = moment_matched_weighted_mean_info_of_doubly_truncated_gaussian
        for i in range(self.weights.shape[0]):
            weighted_mean[i,:,:], info[i,:,:] = moment_match_func(self.weighted_mean[i,:,:], self.info[i,:,:],
                                                             hyperplane_normal, upper_bounds, lower_bounds)

        return GaussianMixtureWeightedMeanInfoMessage(self.weights.copy(), weighted_mean, info)

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return GaussianMixtureWeightedMeanInfoMessage(-self.weighted_mean, self.info)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.allclose(self.weights, other.weights) and\
                   np.allclose(self.weighted_mean, other.weighted_mean) and\
                   np.allclose(self.info, other.info)

        raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, self.__class__):

            assert self.weighted_mean.shape[1:] == other.weighted_mean.shape[1:]
            assert self.info.shape[1:] == other.info.shape[1:]

            n = self.weighted_mean.shape[1]
            num_gaussians_self = self.weights.shape[0]
            num_gaussians_other = other.weights.shape[0]
            num_gaussians = num_gaussians_self * num_gaussians_other

            weights = np.zeros((num_gaussians, 1))
            weighted_mean = np.zeros((num_gaussians, n, 1))
            info = np.zeros((num_gaussians, n, n))

            l = 0
            for i in range(num_gaussians_self):
                for k in range(num_gaussians_other):
                    weights[l] = self.weights[i] * other.weights[k]
                    weighted_mean[l, :, :] = self.weighted_mean[i, :, :] - other.weighted_mean[k, :, :]
                    info[l, :, :] = self.info[i, :, :] - other.info[k, :, :]
                    l += 1

            return GaussianMixtureWeightedMeanInfoMessage(weights, weighted_mean, info)
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.weights.tolist())}, {repr(self.weighted_mean.tolist())}, {repr(self.info.tolist())})"

    def __str__(self):
        return f"Weights:\n{str(self.weights)},\nWeighted Mean:\n{str(self.weighted_mean)},\nInformation Matrix:\n{str(self.info)}"


@inherit_method_docs
class GaussianMixtureTildeMessage(Message):
    """
    For implementation details, refer to:
    * Loeliger, Bruderer et al. (2016): On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing
    * Loeliger et al. (2007): The factor graph approach to model-based signal processing
    * Petersen, Hoffmann, Rostalski (2018): On Approximate Nonlinear Gaussian Message Passing on Factor Graphs
    """

    def __init__(self, xi, W, direction=PortMessageDirection.Undefined):
        raise NotImplementedError
        super().__init__(direction)
        self.xi = col_vec(xi)
        self.W = mat(W)
        assert np.allclose(self.W, self.W.T)
        assert np.ndim(self.xi) == 2 == np.ndim(self.W)
        assert self.xi.shape[0] == self.W.shape[0] == self.W.shape[1]
        assert self.xi.shape[1] == 1

    def combine(self, msg_b, auto_convert=False, try_other=True):
        if isinstance(msg_b, GaussianMeanCovMessage):
            if self.direction == PortMessageDirection.Backward:
                # This (tilde) message is a backwards message, at least from the point of view of the node its exerted
                # from. Hence, we interpret the other message as a forward message, regardless of the directionality of
                # the other port.
                # --> This should enable things like connecting two in ports or two out ports.
                mean = msg_b.mean - msg_b.cov @ self.xi
            elif self.direction == PortMessageDirection.Forward:
                mean = msg_b.mean + msg_b.cov @ self.xi
            else:
                # Don't know anything about the directionality of this message, hence can't determine how to interpret
                # the other message.
                raise RuntimeError('Cannot calculate marginal from tilde message since port directionality unknown.')
            cov = msg_b.cov - msg_b.cov @ self.W @ msg_b.cov
            return GaussianMeanCovMessage(mean, cov, PortMessageDirection.Undefined)
        elif try_other:
            return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError('Combine is not implemented for this data type.')

    def convert(self, target_type, other_msg=None):
        if target_type is type(self):
            return self
        elif other_msg is None:
            raise NotImplementedError('Cannot convert tilde message without information about the other message.')
        elif isinstance(other_msg, GaussianMeanCovMessage):
            n = self.W.shape[0]
            if self == self.non_informative(n):
                # At least one of the fwd and bwd msgs is noninformative
                if other_msg == GaussianMixtureNonInformativeMessage(n):
                    raise RuntimeError('Cannot convert since uninformative.')
                else:
                    return target_type.non_informative(n, direction=self.direction)
            else:
                W_inv = try_inv_else_robinv(self.W)

                cov = W_inv - other_msg.cov
                if self.direction == PortMessageDirection.Forward:
                    mean = W_inv @ self.xi + other_msg.mean
                elif self.direction == PortMessageDirection.Backward:
                    mean = W_inv @ self.xi - other_msg.mean
                else:
                    raise RuntimeError('Cannot convert tilde message of direction is unknown.')
                return GaussianMeanCovMessage(mean, cov, direction=self.direction).convert(target_type)
        else:
            raise NotImplementedError('No message conversion implemented for this other_msg type.')

    def multiply_deterministic(self, matrix, inverse=False):
        if inverse:
            matrix = np.atleast_2d(matrix)
            matrix_h = matrix.transpose().conj()
            xi = matrix_h @ self.xi
            W = matrix_h @ self.W @ matrix
            return GaussianTildeMessage(xi, W)
        else:
            raise NotImplementedError('Forward multiplication is not implemented for this data type.')

    def is_non_informative(self):
        return all(self.W == self.non_informative(self.W.shape[0]).W)

    @staticmethod
    def non_informative(n, direction=PortMessageDirection.Undefined, inf_approx=None):
        xi = np.zeros(n)
        W = np.zeros((n, n))
        if inf_approx is not None:
            for i in range(0, n):
                W[i, i] = 1 / inf_approx
        return GaussianTildeMessage(xi, W, direction=direction)

    def __add__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __radd__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __sub__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __rsub__(self, other):
        return GaussianTildeMessage(self.xi, self.W)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.allclose(self.xi, other.xi) and np.allclose(self.W, other.W)
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return "message.GaussianTildeMessage(" + repr(self.xi.tolist()) + ", " + repr(self.W.tolist()) + ")"

    def __str__(self):
        return "Xi:\n" + str(self.xi) + ",\nW Tilde Matrix:\n" + str(self.W)


@inherit_method_docs
class GaussianMixtureNonInformativeMessage(GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage,
                                    GaussianMixtureTildeMessage):

    def __init__(self, N, direction=PortMessageDirection.Undefined, inf_approx=None):
        Message.__init__(self, direction)
        self.N = N
        nonInformativeMeanCov = GaussianMixtureMeanCovMessage.non_informative(N, inf_approx=inf_approx)
        self.mean = nonInformativeMeanCov.mean
        self.cov = nonInformativeMeanCov.cov
        nonInformativeWeightedMeanInfo = GaussianMixtureWeightedMeanInfoMessage.non_informative(N, inf_approx=inf_approx)
        self.weighted_mean = nonInformativeWeightedMeanInfo.weighted_mean
        self.info = nonInformativeWeightedMeanInfo.info
        nonInformativeTilde = GaussianMixtureTildeMessage.non_informative(N, inf_approx=inf_approx)
        self.xi = nonInformativeTilde.xi
        self.W = nonInformativeTilde.W

    def combine(self, msg_b, auto_convert=False, try_other=True):
        if try_other:
            return msg_b.combine(self, auto_convert=auto_convert, try_other=False)
        else:
            raise NotImplementedError('Combine is not implemented for this data type.')

    def convert(self, target_type):
        return target_type.non_informative(self.N)

    def multiply_deterministic(self, matrix, inverse=False):
        return self

    def is_non_informative(self):
        return True

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return "GaussianNonInformativeMessage(N=" + str(self.N) + ")"

    def __str__(self):
        return "N:\n" + str(self.N)
