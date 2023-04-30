Download Link: https://assignmentchef.com/product/solved-csc411-assignment-5-gaussian-discriminant-analysis
<br>
<ol>

 <li><strong>Gaussian Discriminant Analysis. </strong>For this question you will build classifiers to label images of handwritten digits. Each image is 8 by 8 pixels and is represented as a vector of dimension 64 by listing all the pixel values in raster scan order. The images are grayscale and the pixel values are between 0 and 1. The labels <em>y </em>are 0<em>,</em>1<em>,</em>2<em>,</em>·· <em>,</em>9 corresponding to which character was written in the image. There are 700 training cases and 400 test cases for each digit; they can be found in a2digits.zip.</li>

</ol>

Starter code is provided to help you load the data (data.py). A skeleton (q1.py) is also provided for each question that you should use to structure your code.

Using maximum likelihood, fit a set of 10 class-conditional Gaussians with a separate, full covariance matrix for each class. Remember that the conditional multivariate Gaussian probability density is given by,

(1)

You should take. You will compute parameters <em>µ<sub>kj </sub></em>and Σ<em><sub>k </sub></em>for <em>k </em>∈ (0<em>…</em>9)<em>,j </em>∈

(1<em>…</em>64). You should implement the covariance computation yourself (i.e. without the aid of ’np.cov’). <em>Hint: To ensure numerical stability you may have to add a small multiple of the identity to each covariance matrix. For this assignment you should add </em>0<em>.</em>01<strong>I </strong><em>to each matrix.</em>

<ul>

 <li><strong> </strong>Using the parameters you fit on the training set and Bayes rule, compute the average conditional log-likelihood, i.e. )) on both the train and test set and report it.</li>

 <li><strong> </strong>Select the most likely posterior class for each training and test data point as your prediction, and report your accuracy on the train and test set.</li>

 <li>Compute the leading eigenvectors (largest eigenvalue) for each class covariance matrix (can use linalg.eig) and plot them side by side as 8 by 8 images.</li>

</ul>

Report your answers to the above questions, and submit your completed Python code for q1.py.

<ol start="2">

 <li><strong> Categorial Distribution. </strong>Let’s consider fitting the categorical distribution, which is a discrete distribution over <em>K </em>outcomes, which we’ll number 1 through <em>K</em>. The probability of each category is explicitly represented with parameter <em>θ<sub>k</sub></em>. For it to be a valid probability distribution, we clearly need <em>θ<sub>k </sub></em>≥ 0 and <sup>P</sup><em><sub>k </sub>θ<sub>k </sub></em>= 1. We’ll represent each observation <strong>x </strong>as a 1-of-<em>K </em>encoding, i.e, a vector where one of the entries is 1 and the rest are 0. Under this model, the probability of an observation can be written in the following form:</li>

</ol>

<em>K p</em>(<strong>x</strong>;<em>θ</em>) = Y <em>θ</em><em>kx</em><em>k.</em>

<em>k</em>=1

Denote the count for outcome <em>k </em>as <em>N<sub>k</sub></em>, and the total number of observations as <em>N</em>. In the previous assignment, you showed that the maximum likelihood estimate for the counts was:

<em>.</em>

Now let’s derive the Bayesian parameter estimate.

<ul>

 <li><strong> </strong>For the prior, we’ll use the Dirichlet distribution, which is defined over the set of probability vectors (i.e. vectors that are nonnegative and whose entries sum to 1). Its PDF is as follows:</li>

</ul>

<em>.</em>

A useful fact is that if <em>θ </em>∼ Dirichlet(<em>a</em><sub>1</sub><em>,…,a<sub>K</sub></em>), then

<em>.</em>

Determine the posterior distribution <em>p</em>(<em>θ </em>|D), where D is the set of observations. From that, determine the posterior predictive probability that the next outcome will be <em>k</em>.

<ul>

 <li><strong> </strong>Still assuming the Dirichlet prior distribution, determine the MAP estimate of the parameter vector <em>θ</em>. For this question, you may assume each <em>a<sub>k </sub>&gt; </em></li>

</ul>

<ol start="3">

 <li><strong> Factor Analysis. </strong><em>This question is about the EM algorithm. Since some of you will have seen EM in more detail than others before reading week, we have decided to give you the 4 points for free. So you don’t need to submit a solution to this part if you don’t want to. But we recommend you make an effort anyway, since you probably know enough to solve it, and it will help you practice the course material.</em></li>

</ol>

In lecture, we covered the EM algorithm applied to mixture of Gaussians models. In this question, we’ll look at another interesting example of EM, namely factor analysis. This is a model very similar in spirit to PCA: we have data in a high-dimensional space, and we’d like to summarize it with a lower-dimensional representation. Unlike PCA, we formulate the problem in terms of a probabilistic model. We assume the latent code vector <strong>z </strong>is drawn from a standard Gaussian distribution N(<strong>0</strong><em>,</em><strong>I</strong>), and that the observations are drawn from a diagonal covariance Gaussian whose mean is a linear function of <strong>z</strong>. We’ll consider the slightly simplified case of scalar-valued <em>z</em>. The probabilistic model is given by:

<em>z </em>∼ N(0<em>,</em>1) <strong>x</strong>|<em>z </em>∼ N(<em>z</em><strong>u</strong><em>,</em><strong>Σ</strong>)<em>,</em>

where <strong>Σ </strong>= diag(). Note that the observation model can be written in terms of coordinates: <em>x<sub>j </sub></em>|<em>z </em>∼ N(<em>zu<sub>j</sub>,σ<sub>j</sub></em>)<em>.</em>

We have a set of observations, and <em>z </em>is a latent variable, analogous to the mixture component in a mixture-of-Gaussians model.

In this question, we’ll derive both the E-step and the M-step for the EM algorithm. If you don’t feel like you understand the EM algorithm yet, don’t worry; we’ll walk you through it, and the question will be mostly mechanical.

<ul>

 <li><strong>E-step . </strong>In this step, our job is to calculate the statistics of the posterior distribution <em>q</em>(<em>z</em>) = <em>p</em>(<em>z </em>|<strong>x</strong>) which we’ll need for the M-step. In particular, your job is to find formulas for the (univariate) statistics:</li>

</ul>

<em>m </em>= E[<em>z </em>|<strong>x</strong>] = <em>s </em>= E[<em>z</em><sup>2 </sup>|<strong>x</strong>] =

<em>Tips:</em>

<ul>

 <li>Compare the model here with the linear Gaussian model of the Appendix. Note that <em>z </em>here is a scalar, while the Appendix gives the more general formulation where <strong>x </strong>and <strong>z </strong>are both vectors.</li>

 <li>Determine <em>p</em>(<em>z </em>|<strong>x</strong>). To help you check your work: <em>p</em>(<em>z </em>|<strong>x</strong>) is a univariate Gaussian distribution whose mean is a linear function of <strong>x</strong>, and whose variance does not depend on <strong>x</strong>.</li>

 <li>Once you have figured out the mean and variance, that will give you the conditional expectations.</li>

</ul>

<ul>

 <li><strong>M-step. </strong>In this step, we need to re-estimate the parameters of the model. The parameters are <strong>u </strong>and Σ = diag(). For this part, your job is to derive a formula for <strong>u</strong><sub>new </sub>that maximizes the expected log-likelihood, i.e.,</li>

</ul>

<strong>u</strong><sub>new </sub>← argmax<em>. </em><strong>u</strong>

(Recall that <em>q</em>(<em>z</em>) is the distribution computed in part (a).) This is the new estimate obtained by the EM procedure, and will be used again in the next iteration of the E-step. Your answer should be given in terms of the <em>m</em><sup>(<em>i</em>) </sup>and <em>s</em><sup>(<em>i</em>) </sup>from the previous part. (I.e., you don’t need to expand out the formulas for <em>m</em><sup>(<em>i</em>) </sup>and <em>s</em><sup>(<em>i</em>) </sup>in this step, because if you were implementing this algorithm, you’d use the values <em>m</em><sup>(<em>i</em>) </sup>and <em>s</em><sup>(<em>i</em>) </sup>that you previously computed.)

<em>Tips:</em>

<ul>

 <li>Expand log<em>p</em>(<em>z</em><sup>(<em>i</em>)</sup><em>,</em><strong>x</strong><sup>(<em>i</em>)</sup>) to log<em>p</em>(<em>z</em><sup>(<em>i</em>)</sup>)+log<em>p</em>(<strong>x</strong><sup>(<em>i</em>) </sup>|<em>z</em><sup>(<em>i</em>)</sup>) (log is the natural logarithm).</li>

 <li>Expand out the PDF of the Gaussian distribution.</li>

 <li>Apply linearity of expectation. You should wind up with terms proportional to E<em><sub>q</sub></em><sub>(<em>z</em></sub>(<em><sub>i</sub></em><sub>))</sub>[<em>z</em><sup>(<em>i</em>)</sup>] and E<em><sub>q</sub></em><sub>(<em>z</em></sub>(<em><sub>i</sub></em><sub>)</sub>[[<em>z</em><sup>(<em>i</em>)</sup>]<sup>2</sup>]. Replace these expectations with <em>m</em><sup>(<em>i</em>) </sup>and <em>s</em><sup>(<em>i</em>)</sup>. You should get an equation that does not mention <em>z</em><sup>(<em>i</em>)</sup>.</li>

 <li>In order to find the maximum likelihood parameter <strong>u</strong><sub>new</sub>, you need to take the derivative with respect to <em>u<sub>j</sub></em>, set it to zero, and solve for <strong>u</strong><sub>new</sub>.</li>

</ul>

<ul>

 <li><strong>M-step, cont’d (optional) </strong>Find the M-step update for the observation variances . This can be done in a similar way to part (b).</li>

</ul>

<h1>Appendix: Some Properties of Conditional Gaussians</h1>

Consider a multivariate Gaussian random variable <strong>z </strong>with the mean <em>µ </em>and the covariance matrix Λ<sup>−1 </sup>(Λ is the inverse of the covariance matrix and is called the precision matrix). We denote this by

<em>p</em>(<strong>x</strong>) = N(<strong>z</strong>|<em>µ,</em>Λ<sup>−1</sup>)<em>.</em>

Now consider another Gaussian random variable <strong>x</strong>, whose mean is an affine function of <strong>z </strong>(in the form to be clear soon), and its covariance <em>L</em><sup>−1 </sup>is independent of <strong>z</strong>. The conditional distribution of <strong>x </strong>given <strong>z </strong>is

<em>p</em>(<strong>x</strong>|<strong>z</strong>) = N(<strong>x</strong>|<em>A</em><strong>z </strong>+ <em>b,L</em><sup>−1</sup>)<em>.</em>

Here the matrix <em>A </em>and the vector <em>b </em>are of appropriate dimensions.

In some problems, we are interested in knowing the distribution of <strong>z </strong>given <strong>x</strong>, or the marginal distribution of <strong>x</strong>. One can apply Bayes’ rule to find the conditional distribution <em>p</em>(<strong>z</strong>|<strong>x</strong>). After some calculations, we can obtain the following useful formulae:

with

<em>C </em>= (Λ + <em>A</em><sup>&gt;</sup><em>LA</em>)<sup>−1</sup><em>.</em>