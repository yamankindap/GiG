
function [x_gamma,gamma_var]=gen_gamma_process(M,gamma_param,C)

% Generates M points from the process Cx^{-1}e^{-gamma_param x}
exp_rnd=exprnd(1,M,1);
gammas=cumsum(exp_rnd);

% Generate from envelope process
x=1./(gamma_param*(exp(gammas./C)-1));

accept_prob=(1+x*gamma_param).*exp(-x*gamma_param);
rand_unif=rand(length(accept_prob),1);

x_gamma=x(rand_unif<accept_prob);

gamma_var=sum(x_gamma);