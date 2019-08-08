% @Author: Xiaoyu He
% @Date:   2018-05-26 20:39:01
% @Last Modified by:   hxy
% @Last Modified time: 2019-08-08 22:18:33

% X. He, Y. Zhou, Z. Chen, J. Zhang, and W. Chen, Large-Scale Evolution Strategy Based on Search Direction Adaptation, IEEE Transactions on Cybernetics, in press, 2019.

% search direction adaptation ES

% fobj: object function
% dim: dimensionality
% opt: parameter options
%   opt.x0: starting point
%   opt.sigma0: initial mutation strength

%   opt.stopfitness: stopping criterion --- target fitness value
%   opt.maxFEs: stopping criterion --- maximum number of function evaluations
%   opt.stopsigma: stopping criterion --- minimum mutation strength
%   opt.stoptime: stopping criterion --- maximum running time

%   opt.verbose: displaying message

function out = SDAES(fobj, dim, opt)
xmean = opt.x0;
sigma = opt.sigma0;
stopfitness = opt.stopfitness;
maxFEs = opt.maxFEs;
stopsigma = opt.stopsigma;
stoptime = opt.stoptime;

% parameters from CMA-ES
lambda = (4+floor(3*log(dim)));     
mu = floor(lambda/2);
weights = log(mu+1/2)-log(1:mu)';
weights = weights/sum(weights);
mueff=1/sum(weights.^2);

% number of search directions
m = 10;
% learning rate for covariance matrix adaptation
ccov = 0.4/sqrt(dim);
% learning rate for search direction adaptation
cc = 0.25/sqrt(dim);

%% parameters for mutation strength adaptation
len = 1 : 2*lambda;  
% accumulated Z statistic
Zcum = 0;  
% damping factor
ds = 1;
% learning rate 
cs = 0.3;   
% target significant level
qtarget = 0.05;  

% elitist fitness
fbest = fobj(xmean);
% fitness values of the previous population
prevfits = fbest*ones(1, lambda);   
% matrix containing the search directions 
Q = randn(dim,m)*1e-6;

% ------------------- Generation Loop --------------------------------
FEs = 1;  
gen = 1; 
% record interval
recordGap = 1e3;
record = zeros(ceil(maxFEs/lambda/recordGap+2e3/lambda),2);
recordIter = 1;

inTic = tic;
while fbest > stopfitness && FEs < maxFEs && sigma > stopsigma && toc(inTic) <= stoptime
    % mirroring for fast sampling
    arY = sqrt(1-ccov) * randn(dim,ceil(lambda/2)) ...
        + sqrt(ccov) * Q * randn(m,ceil(lambda/2));
    arX = xmean + sigma*[arY -arY(:,1:floor(lambda/2))];
    
    arFitness = fobj(arX); 
    FEs = FEs+lambda;

    % sort
    [arFitness, arIndex] = sort(arFitness);
    xold = xmean;
    arIndex = arIndex(1:mu);

    % recombination
    xmean = arX(:,arIndex) * weights;  
    % success mutation step
    z = sqrt(mueff) * (xmean-xold) / sigma;
   
    % search direction adaptation
    for i = 1 : m
        % update the search direction
        Q(:,i) = (1-cc) * Q(:,i) + sqrt(cc*(2-cc))*z;
        % scaled projection length
        t = z'*Q(:,i)/(Q(:,i)'*Q(:,i));
        % normalized residual between the success mutation step and the updated search direction
        z = (z - t*Q(:,i))/sqrt(1+t^2);
    end

    %% adapt step-size
    % merge the current and previous populations and sort
    [~, imax] = sort([prevfits, arFitness(1:lambda)]);
    % rank sum of previous population
    R1 = sum(len(imax<=lambda));
    % U test
    U1 = R1 - lambda*(lambda+1)/2;
    % de-randomization: U1 has a distribution whose mean is (lambda^2/2) and variance is (lambda^2*(2*lambda+1)/12)
    W = U1 - lambda^2/2;
    W = W / sqrt(lambda^2*(2*lambda+1)/12);
    % W is exactly the Z statistic, i.e., the output of the U test, so we further accumulate it:
    Zcum = (1-cs) *Zcum + sqrt(cs*(2-cs))*W;
    % test the significant level for Zcum and keep it not far away from qtarget
    sigma = sigma * exp((normcdf(Zcum)/(1-qtarget)-1)/ds); 
    prevfits = arFitness(1:lambda);
    
    % recording
    fbest = min(fbest, arFitness(1));
    if mod(gen,recordGap) == 1 
        record(recordIter,:) = [FEs fbest];
        if opt.verbose > 0
            fprintf('#%d FEs=%d fit=%g\n', recordIter, FEs, fbest);
        end
        recordIter = recordIter + 1;
    end
    gen = gen + 1;  
end
record(recordIter,:) = [FEs fbest];
if size(record,1) > recordIter
    record(recordIter+1:end,:) = [];
end
out.record = record;
out.bestFitness = fbest;