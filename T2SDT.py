"""
Calculate the type 2 Signal Detection Theory (SDT) measure meta-d' for the
S1 and S2 responses according to the method described in:

Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach
for estimating metacognitive sensitivity from confidence ratings.
Consciousness and Cognition, 21(1), 422-430.
doi:10.1016/j.concog.2011.09.021

and

Maniscalco, B., & Lau, H. (2014). Signal detection theory analysis of
type 1 and type 2 data: meta-d', response-specific meta-d', and the
unequal variance SDT mode. In S. M. Fleming & C. D. Frith (Eds.),
The Cognitive Neuroscience of Metacognition (pp.25-66). Springer.

Only the equal variance approach and normally distributed inner decision
variables are currently supported. The function calculates the
response-specific meta-d' variables.

The performance of this code was compared to the Matlab code available
at http://www.columbia.edu/~bsm2105/type2sdt/
Results were equivalent. However, this Python code was about 15x faster.

Usage:
------
The class T2SDT implements the optimization of the type 2 SDT model.
As data, a confusion matrix (including confidence ratings) should be given.

After initialization, the fit() method of the class can be used to fit
the type 2 SDT model to the supplied data.

The confusion matrix (including condidence ratings) can be calculated
from data using the function confusion_matrix
"""

from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

# ignore division by 0 and invalid numerical operations
np.seterr(invalid='ignore', divide='ignore')

class T2SDT(object):
    """ Fit a type 2 SDT analysis to the provided confusion matrix.

    Attributes:
        conf_matrix (array of ints): the confusion matrix, taking into
            account the confidence ratings (cf. documentation to
            confusion_matrix function), might be adjusted to avoid
            fields containing zero (depends on the adjust parameter)
        max_conf (int>0): the maximal possible confidence rating
        HR (float): the type 1 Hit Rate
        FAR (float): the type 1 False Alarm Rate
        d (float): the d' measure for the type 1 classification task
            denotes the ability to discriminate between S1 (stim. absent)
            and S2 (stim present) trials in standard deviation units
        c (float): the response bias of the type 1 classification task
        meta_d_S1 (float): the meta-d' measure for S1 responses
            denotes the ability of the subject to discriminate between
            correct and incorrect S1 responses in the same units as those of
            the type 1 task, only available after calling the fit method
        meta_d_S2 (float): the meta-d' measure for S2 responses
            denotes the ability of the subject to discriminate between
            correct and incorrect S2 responses in the same units as those of
            the type 1 task, only available after calling the fit method
        meta_c_S1 (float): the response bias of the type 2 task for S1
            responses
            meta_c_S1 is calculated such that meta_c_S1' = c', where
            c' = c/d' and meta_c_S1' = meta_c_S1/meta_d_S1'
            only available after calling the fit method
        meta_c_S2 (float): the response bias of the type 2 task for S2
            responses
            meta_c_S2 is calculated such that meta_c_S2' = c', where
            c' = c/d' and meta_c_S2' = meta_c_S2/meta_d_S2'
            only available after calling the fit method
        meta_c2_S1 (array of floats): the decision criteria for each
            confidence level if the response had been "S1" (stim. absent)
            meta_c2_S1[0] belongs to confidence==1
            meta_c2_S1[0] belongs to confidence==2
            etc.
            meta_c2_S1[0]>=meta_c2_S1[1]>=meta_c2_S1[2] ... >= meta_c_S1
            only available after calling the fit method
        meta_c2_S2 (array of floats): the decision criteria for each
            confidence level if the response had been "S2" (stim. present)
            meta_c2_S2[0] belongs to confidence==1
            meta_c2_S2[0] belongs to confidence==2
            etc.
            meta_c2_S2[0]<=meta_c2_S2[1]<=meta_c2_S2[2] ... <= meta_c_S2
            only available after calling the fit method
        logL_S1 (float): the log-likelihood of the model for S1 responses;
            only available after calling the fit method
        logL_S2 (float): the log-likelihood of the model for S2 responses;
            only available after calling the fit method
        success_S1 (bool): whether the fitting was successful for S1
            responses; only available after calling the fit method
        success_S2 (bool): whether the fitting was successful for S2
            responses; only available after calling the fit method
        fit_message_S1 (str): the message output by the optimization
            algorithm for S1 responses; only available after calling the
            fit method
        fit_message_S2 (str): the message output by the optimization
            algorithm for S2 responses; only available after calling the
            fit method
    """

    def __init__(self, conf_matrix, adjust=False):
        """
        Args:
            conf_matrix (array of ints, shape 2x2x(max_conf + 1)): the
                confusion matrix including confidence ratings
                - conf_matrix[0,:] contains all trials that were actually
                    negative and conf_matrix[1,:] all trials that were
                    actually positive
                - conf_matrix[:,0] contains all trials that were classified
                    as being negative and conf_matrix[:,1] all trials that
                    were classified as being positive
                Accordingly:
                - TN are in conf_matrix[0,0]
                - TP are in conf_matrix[1,1]
                - FN are in conf_matrix[1,0]
                - FP are in conf_matrix[0,1]
                The last axis is determined by the confidence rating:
                - in conf_matrix[...,0], rating was 0
                - in conf_matrix[...,1], rating was 1 
                - in conf_matrix[...,2], rating was 2
            adjust (bool): if True, 1./(2*(max_conf + 1)) is added to all
                entries of the confusion matrix to avoid zero entries
        """
        conf_matrix = np.asarray(conf_matrix)
        if not conf_matrix.ndim==3:
            raise ValueError('conf_matrix must be 3d')
        if not conf_matrix.shape[:2] == (2,2):
            raise ValueError('the shape of the conf_matrix must be'
                    ' 2x2xn, where n is the number of possible confidence'
                    ' ratings.')
        self.conf_matrix = conf_matrix
        self.max_conf = self.conf_matrix.shape[-1] - 1
        if self.max_conf < 1:
            raise ValueError('All confidence ratings are equal.')
        if adjust:
            self.conf_matrix = self.conf_matrix + 1./(2*(self.max_conf + 1))
        if np.any(self.conf_matrix == 0):
            warnings.warn('Some entries of the confusion matrix are 0'
            ' This might cause problems with fitting the SDT model.',
            UserWarning)
        TP = self.conf_matrix.sum(-1)[1,1] # true positives
        FN = self.conf_matrix.sum(-1)[1,0] # false negatives
        TN = self.conf_matrix.sum(-1)[0,0] # true negatives
        FP = self.conf_matrix.sum(-1)[0,1] # false positives
        self.HR = TP/float(TP + FN) # the hit rate
        self.FAR = FP/float(FP + TN) # the false alarm rate
        z_HR = norm.ppf(self.HR)
        z_FAR = norm.ppf(self.FAR)
        self.d = z_HR - z_FAR # the type I sensitivity
        self.c = -0.5*(z_HR + z_FAR) # the type I decision criterion

    def fit(self):
        """
        Fit the type 2 SDT model to maximize log-likelihood between the
        model and the obervations.
        This generates the attributes meta_d_S1, meta_c_S1, meta_c_S2,
        meta_c2_S1, meta_c2_S2
        """
        ###############################
        # fit meta_d for S1 responses #
        ###############################
        result_S1 = minimize(self._get_log_likelihood,
                x0 = [0] + self.max_conf*[-0.1],
                args = ('S1',),
                method = 'L-BFGS-B',
                jac = True,
                bounds = ([(None,None)] + self.max_conf*[(None, 0)]),
                options=dict(disp=False))
        self.meta_d_S1, self.meta_c_S1, self.meta_c2_S1 = (
                self._get_parameters(result_S1.x))
        self.logL_S1 = -result_S1.fun
        self.success_S1 = result_S1.success
        self.fit_message_S1 = result_S1.message
        ###############################
        # fit meta_d for S2 responses #
        ###############################
        result_S2 = minimize(self._get_log_likelihood,
                x0 = [0] + self.max_conf*[0.1],
                method = 'L-BFGS-B',
                args = ('S2',),
                jac = True,
                bounds = ([(None,None)] + self.max_conf*[(0, None)]),
                options=dict(disp=False))
        self.meta_d_S2, self.meta_c_S2, self.meta_c2_S2 = (
                self._get_parameters(result_S2.x))
        self.logL_S2 = -result_S2.fun
        self.success_S2 = result_S2.success
        self.fit_message_S2 = result_S2.message

    def _get_log_likelihood(self, x, which='S1', return_der=True):
        """Internal method, do not use directly!

        Calculates the (negative) log-likelihood of the fitted model.
        The negative log-likelihood is returnd to maximize the
        log-likelihood by minimizing the output of this function.
        
        The docstring to the method _get_parameters explains how x is
        translated to the parameters of the type 2 model
        """
        if not which in ['S1', 'S2']:
            raise ValueError('which must be S1 or S2')
        meta_d, meta_c, meta_c2 = self._get_parameters(x)
        # initialize an empty matrix of probabilities for each outcome
        # (i.e., for each combination of stimulus, response and confidence)
        cumprobs = np.empty([2, self.max_conf + 2])
        probs = np.empty([2,self.max_conf + 1], float)
        ##########################################
        # calculate the elementary probabilities #
        ##########################################
        # calculate the response-specific cumulative probabilities for all
        # ratings
        if which is 'S1':
            cumprobs[0] = np.r_[norm.cdf(np.r_[meta_c, meta_c2],
                -0.5*meta_d),0]
            cumprobs[1] = np.r_[norm.cdf(np.r_[meta_c, meta_c2],
                0.5*meta_d), 0]
        else:
            cumprobs[0] = np.r_[norm.sf(np.r_[meta_c, meta_c2],
                -0.5*meta_d), 0]
            cumprobs[1] = np.r_[norm.sf(np.r_[meta_c, meta_c2],
                0.5*meta_d), 0]
        # calculate the response-specific probabilities for all ratings
        probs = (cumprobs[...,:-1] - cumprobs[...,1:])/cumprobs[...,0,
                np.newaxis]
        # calculate the log likelihood
        if which is 'S1':
            total_logp = np.sum(np.log(probs)*self.conf_matrix[:,0], None)
        else:
            total_logp = np.sum(np.log(probs)*self.conf_matrix[:,1], None)
        if return_der:
            # calculate the derivative
            total_logp_der = np.zeros(len(x), float)
            # calculate the derivative of cumprobs
            cumprobs_der = np.zeros([2,self.max_conf + 2])
            if which is 'S1':
                cumprobs_der[0] = np.r_[norm.pdf(np.r_[
                    meta_c, meta_c2], -0.5*meta_d), 0]
                cumprobs_der[1] = np.r_[norm.pdf(np.r_[
                    meta_c, meta_c2],  0.5*meta_d), 0]
            else:
                cumprobs_der[0] = np.r_[-norm.pdf(np.r_[
                    meta_c, meta_c2], -0.5*meta_d), 0]
                cumprobs_der[1] = np.r_[-norm.pdf(np.r_[
                    meta_c, meta_c2], 0.5*meta_d), 0]
            #################################################
            # calculate derivatives for the meta-d' element #
            #################################################
            cumprobs_der_meta_d = cumprobs_der.copy()
            cumprobs_der_meta_d[0] *= self.c/self.d + 0.5
            cumprobs_der_meta_d[1] *= self.c/self.d - 0.5
            # calculate the derivative of probs according to the quotient
            # rule
            probs_der_meta_d = (
                    (cumprobs_der_meta_d[...,:-1] -
                        cumprobs_der_meta_d[...,1:])*cumprobs[
                            ...,0,np.newaxis] -
                        (cumprobs[...,:-1] - cumprobs[...,1:])*
                        cumprobs_der_meta_d[...,0,np.newaxis])/(
                                cumprobs[...,0,np.newaxis]**2)
            if which is 'S1':
                total_logp_der[0] = np.sum(
                        self.conf_matrix[:,0]/probs*probs_der_meta_d, None)
            else:
                total_logp_der[0] = np.sum(
                        self.conf_matrix[:,1]/probs*probs_der_meta_d, None)
            ############################################
            # calculate derivative for the c2 elements #
            ############################################
            cumprobs_der /= cumprobs[...,0, np.newaxis]
            cumprobs_der_diff = cumprobs_der[...,:-1] - cumprobs_der[...,1:]
            # calculate the derivative of the log od the probs and the
            # product with the confidence ratings
            if which is 'S1':
                log_cumprobs_der = (
                        cumprobs_der[...,1:]*self.conf_matrix[:,0]/probs)
                log_cumprobs_diff_der = cumprobs_der_diff[
                    ...,1:]*self.conf_matrix[...,0,1:]/probs[...,1:]
            else:
                log_cumprobs_der = (
                        cumprobs_der[...,1:]*self.conf_matrix[:,1]/probs)
                log_cumprobs_diff_der = cumprobs_der_diff[
                    ...,1:]*self.conf_matrix[...,1,1:]/probs[...,1:]
            total_logp_der[1:] = (
                    np.cumsum(
                        log_cumprobs_diff_der.sum(0)[...,::-1], axis=-1)[
                            ...,::-1] - log_cumprobs_der[
                                ...,:self.max_conf].sum(0))
            return -total_logp, -total_logp_der
        else:
            return -total_logp

    def _get_parameters(self, x):
        """Internal method, do not use directly!

        From the optimization input x, get meta_d, meta_c_S1, and meta_c_S2

        The passed parameter list x consists of
         - meta_d
         - the offsets x2 such that:
             meta_c2 = meta_c + np.cumsum(x2)
             The length of x2 must be max_conf.
             - If x2 is strictly negative, this results in the meta_c2
             parameters for S1.
             - If x2 is strictly positive, this results in the meta_c2
             parameters for S2.

        Meta_c is chosen such that meta_c/meta_d' = c/d'

        Notes:
            Let d' and c of the primary condition be d' = 2 and c = 1
            max_conf = 2 (i.e., confidence is rated on a scale 0-1-2)

            If a parameter list x = [0.7, -0.1, -0.05] is passed,
            this leads to the following arguments:

            meta_c = c/d' * meta_d = 2/1 * 0.7 = 1.4
            meta_c2_S1 = 1.4 + cumsum([-0.1, -0.05]) = [1.3, 1.25]
            
            If a parameter list x = [0.7, 0.1, 0.05] is passed,
            this leads to the following arguments:

            meta_c = c/d' * meta_d = 2/1 * 0.7 = 1.4
            meta_c2_S2 = 1.4 + cumsum([0.1, 0.05]) = [1.5, 1.55]
        """
        if not len(x) == self.max_conf + 1:
            raise TypeError('length of x does not fit the expected length')
        meta_d = x[0]
        meta_c = self.c/self.d*meta_d
        meta_c2 = meta_c + np.cumsum(x[1:])
        return meta_d, meta_c, meta_c2

def confusion_matrix(true_label, pred_label, rating, max_conf=None):
    """
    Calculate a 2x2x(max_conf + 1) confusion matrix.

    Args:
        true_label (array of ints): the actual stimulus condition
            should be 0 for an absent stimulus and 1 for a present
            stimulus
        pred_label (array of ints): the predicted stimulus condition
            should be 0 if the stimulus was classified as being present
            and 1 if the stimulus was classified as being absent
        rating (array of ints, rating >= 0): confidence rating on an 
            ordered integer scale (0, 1, ..., max_conf) where 0 means
            low confidence and max_conf means maximal confidence
        max_conf (int>=0, optional): the maximal possible confidence
            value, if not given the maximal value of the given
            rating is chosen. E.g., if max_conf=2, 3 possible confidence
            levels are associated to the classification task
            0 - unsure, 1 - neutral, 2 - sure

    Returns:
        conf_matrix (array of ints, shape 2x2x(max_conf + 1)): the confusion
            matrix
            - conf_matrix[0,:] contains all trials that were actually
                negative
            - conf_matrix[:,0] contains all trials that were classified as 
                being negative
            Accordingly:
            - TN are in conf_matrix[0,0]
            - TP are in conf_matrix[1,1]
            - FN are in conf_matrix[1,0]
            - FP are in conf_matrix[0,1]
            The last axis is determined by the confidence rating:
            - in conf_matrix[...,0], rating was 0
            - in conf_matrix[...,1], rating was 1 
            - in conf_matrix[...,2], rating was 2

    Note:
        An "ordinary" confusion matrix (i.e., without taking confidence
        into account) is obtained as conf_matrix.sum(axis=-1)
    """
    ###################
    # variable checks #
    ###################
    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    if not np.allclose(true_label, true_label.astype(int)):
        raise TypeError('all labels must be integers')
    if not np.allclose(pred_label, pred_label.astype(int)):
        raise TypeError('all labels must be integers')
    true_label = true_label.astype(int)
    pred_label = pred_label.astype(int)
    if not np.all([true_label>=0, true_label<=1]):
        raise ValeError('all labels must be 0 or 1')
    if not np.all([pred_label>=0, pred_label<=1]):
        raise ValeError('all labels must be 0 or 1')
    #
    rating = np.asarray(rating)
    if not np.allclose(rating, rating.astype(int)):
        raise TypeError('all ratings must be integers')
    rating = rating.astype(int)
    if not np.all(rating >= 0):
        raise ValueError('all ratings must be equal to or larger than 0')
    if max_conf is None:
        max_conf = rating.max()
    else:
        if not type(max_conf) == int:
            raise TypeError('max_conf must be an integer')
        if max_conf < 0:
            raise ValueError('max_conf must be >= 0')
        if not np.all(rating <= max_conf):
            raise ValueError('all ratings must be smaller than or equal'
                    ' to max_conf')
    ##################################
    # calculate the confusion matrix #
    ##################################
    conf_matrix = np.zeros([2,2,max_conf + 1], int)
    for true_now, pred_now, rating_now in zip(true_label, pred_label,
            rating):
        conf_matrix[true_now, pred_now, rating_now] += 1
    return conf_matrix

if __name__ == "__main__":
    ####################################
    # Simulate a binary detection task #
    ####################################
    d = 0.6 # d_prime of type 1 task
    c = 0 # c of type 1 task
    N1 = 400 # number of absent stimuli
    N2 = 800 # number of present stimuli
    err = 1 # standard deviation of noise
    c_S1 = [0.34, 1.1] # criteria for confidence in case response S1
    c_S2 = [1.6, 1.9] # criteria for confidence in case response S2
    true_label = np.r_[[0]*N1, [1]*N2]
    x = np.r_[norm.rvs(-0.5*d, size=N1), norm.rvs(0.5*d, size=N2)]
    pred_label = np.where(x<c, 0, 1)
    # x2 is a noisy version of x and used to calculate confidence criteria
    x2 = x + norm.rvs(scale=err, size=len(x))
    # create the confidence ratings
    confidence = np.zeros(x.shape, int)
    for k,c_now in enumerate(c_S1):
        confidence[pred_label == 0] = np.where(
                x2[pred_label==0] < (c - c_now), k + 1,
                confidence[pred_label==0])
    for k, c_now in enumerate(c_S2):
        confidence[pred_label == 1] = np.where(
                x2[pred_label==1] >= (c + c_now), k + 1,
                confidence[pred_label == 1])
    #################################
    # Now, fit the type 2 SDT model #
    #################################
    # calculate the confusion matrix
    conf_matrix = confusion_matrix(true_label, pred_label, confidence,
            max_conf=2)
    model = T2SDT(conf_matrix, adjust=False)
    model.fit()
    print('Results of the simuluated type 2 SDT')
    print('------------------------------------')
    print('d\': %.2f, c: %.2f' % (model.d, model.c))
    print('------------------------------------')
    print('S1 model fitting success: %s' % model.success_S1)
    print('S1 model fitting message:\n    %s' % model.fit_message_S1)
    print('meta-d_S1\': %.2f, meta-c_S1: %.2f, logL_S1: %.2f' % (
            model.meta_d_S1, model.meta_c_S1, model.logL_S1))
    print('------------------------------------')
    print('S2 model fitting success: %s' % model.success_S2)
    print('S2 model fitting message:\n    %s' % model.fit_message_S2)
    print('meta-d_S2\': %.2f, meta-c_S2: %.2f, logL_S2: %.2f' % (
            model.meta_d_S2, model.meta_c_S2, model.logL_S2))
