clear all
close all
gamma_param=0.1;
lambda=-3;
delta=3;

N_samps=10000;

[GIG_var,GIG_jumps_store,jump_times_store,Cumulative_jumps_store]=gig_gen(gamma_param,lambda,delta,N_samps);

disp('Finished: now generating plots')

if (gamma_param>0)
gig_vars=gigrnd(lambda, gamma_param^2, delta^2,  N_samps);
else
gig_vars=1./gamrnd(-lambda,2/delta^2,N_samps,1);
end

subplot(121),
qqplot(GIG_var,gig_vars)
title('Sample QQ plot')
xlabel('GIG shot noise process, t=1')
ylabel('GIG random variable')


[y_gig,x_gig]=hist(GIG_var,1000);
   
[y_gig_true,x_gig]=hist(gig_vars,x_gig);


% Compute GiG density function:
if (gamma_param>0)
   gig_density=x_gig.^(lambda-1).*exp(-0.5*(delta^2./x_gig+gamma_param^2*x_gig)).*(gamma_param/delta)^lambda./(2*besselk(lambda,delta*gamma_param));
else
   gig_density=x_gig.^(lambda-1).*exp(-0.5*(delta^2./x_gig)).*(0.5*delta^2)^(-lambda)./gamma(-lambda);
end
subplot(122),
loglog(x_gig(2:end-1),y_gig(2:end-1)./((x_gig(2)-x_gig(1))*(length(GIG_var))),x_gig(1:end-1),gig_density(1:end-1));%y_Sig(1:end-1));%,x_gig(1:end-1),y_TS(1:end-1))
title('Histogram comparison')

legend('GIG shot noise process, t=1','GIG probability density')
xlabel('x')

figure
% Log histograms:

subplot(121),
qqplot(log(GIG_var),log(gig_vars))
title('Sample log-log QQ plot')
xlabel('GIG shot noise process, t=1')
ylabel('GIG random variable')


% These are plots for the log-GIG variables, works better for very
% heavy-tailed parameter settings:
[y_gig,x_gig]=hist(log(GIG_var),100);
   
[y_gig_true,x_gig]=hist(log(gig_vars),x_gig);



% Compute log-GiG density function (introduces a Jacobian factor of e^x in the transformed density:
x_gig_exp=exp(x_gig);
if (gamma_param>0) 
   gig_density=x_gig_exp.^(lambda).*exp(-0.5*(delta^2./x_gig_exp+gamma_param^2*x_gig_exp)).*(gamma_param/delta)^lambda./(2*besselk(lambda,delta*gamma_param));
else
   gig_density=x_gig_exp.^(lambda).*exp(-0.5*(delta^2./x_gig_exp)).*(0.5*delta^2)^(-lambda)./gamma(-lambda);
end
subplot(122),
semilogy(x_gig(2:end-1),y_gig(2:end-1)./((x_gig(2)-x_gig(1))*(length(GIG_var))),x_gig(1:end-1),gig_density(1:end-1));%y_Sig(1:end-1));%,x_gig(1:end-1),y_TS(1:end-1))
title('Histogram comparison - log densities')

legend('GIG shot noise process, t=1','GIG probability density')
xlabel('x')

hold off

figure
hold on
% Plot a selection of jump paths (very slow if you plot them all!!)
for q=1:1000:length(GIG_var)
plot(jump_times_store{q}(:),Cumulative_jumps_store{q}(:))
title('Path of GIG process')
xlabel('t')
ylabel('X(t)')
end
hold off


function [GIG_var,GIG_jumps_store,jump_times_store,Cumulative_jumps_store]=gig_gen(gamma_param,lambda,delta,N_samps)

% Generate GIG random process

% Input Parameters:
% gamma_param>=0 
% lambda \in \Re
% delta>0

% N_samps - integer number of independent samples to draw

% Outputs:
% GIG_var - random samples from the GIG process at t=1
% GIG_jumps_store - cell array: jump values for each GIG_var sample
% jump_times_store - cell array: times of jumps for """"
% Cumulative_jumps_store - cell array: cumulative process for """""


% Warning!! Note the non-standard Matlab definition of the incomplete gamma
% function! Defined to lie between 0 and 1

disp('Generating GIG sample number:')

% For |lambda|>0.5 series:
K=delta*gamma(0.5)/(sqrt(2)*pi);

% For |lambda|<0.5 series:
z_0=((2^(1-2*abs(lambda))*pi)/(gamma(abs(lambda))^2))^(1/(1-2*abs(lambda)));
H_0=z_0*(bessely(abs(lambda),(z_0)).^2+besselj(abs(lambda),[(z_0)]).^2);

K1=(2*delta^2)^abs(lambda)/(pi^2*H_0*z_0^(2*abs(lambda)-1))*gamma(abs(lambda));
K2=(2*delta^2)^0.5/(pi^2*H_0)*gamma(0.5);

% Number of terms in truncated series:
M=1000;

accept_rate1=0;
accept_rate1a=0;
accept_rate2=0;
accept_rate2a=0;

% Which sampling method for point process N1:
method=2;
if (gamma_param==0)
    method=1;
end    

for q=1:N_samps

if mod((q-1),1000)==0
    q
end

if (abs(lambda)>0.5)

exp_rnd=exprnd(1,M,1);
gammas=cumsum(exp_rnd);

% Generate positive stable variable, alpha=0.5:
x=(gammas./(2*K)).^(-2);

accept_prob=exp(-x*gamma_param^2./2);
rand_unif=rand(length(accept_prob),1);

% Generate tempered stable:
x_TS=x(rand_unif<accept_prob);

% Generate auxiliary variable:
y=gamrnd(0.5,1./x_TS);

unif=rand(length(y),1);

accept_prob=(sqrt(2)./(delta*pi*y.^(0.5).*(bessely(abs(lambda),delta*sqrt(2*y)).^2+besselj(abs(lambda),delta*sqrt(2*y)).^2)));
accept=unif<accept_prob;

% Generate GIG variable:
GIG_var(q)=sum(x_TS(accept));
GIG_var2(q)=GIG_var(q);
GIG_var1(q)=0*GIG_var(q);

GIG_points=x_TS(accept);

% Generate tempered stable variate:
TS_var(q)=sum(x_TS);


else
    
if (method==1)    
exp_rnd1=exprnd(1,M,1);
gammas1=cumsum(exp_rnd1);

% Generate positive stable variable, alpha=|lambda|:
x1=(gammas1./(K1/abs(lambda))).^(-1/abs(lambda));

accept_prob1=exp(-x1*gamma_param^2./2).*gammainc(z_0^2*x1./(2*delta^2),abs(lambda));%./gamma(abs(lambda));
rand_unif=rand(length(accept_prob1),1);

else
    
C1=1/(pi^2*H_0*abs(lambda)*z_0^(-1));%*(1+abs(lambda)))    
x1=gen_gamma_process(M,gamma_param^2/2,C1);    

% Get rid of any numerically zero terms:
x1=x1(x1>0);

accept_prob1=abs(lambda)*gammainc(z_0^2*x1./(2*delta^2),abs(lambda)).*gamma(abs(lambda)).*((2*delta^2)./(z_0^2*x1)).^abs(lambda);
% Really tiny jumps accepted: asymptotic version of gammainc at x->0:
accept_prob1(gammainc(z_0^2*x1./(2*delta^2),abs(lambda))==0)=1;
accept_prob1=min(1,accept_prob1);
rand_unif=rand(length(accept_prob1),1);

end    
accept_rate1=mean(accept_prob1)/q+accept_rate1*(q-1)/(q);

% Generate tempered stable:
x_TS1=x1(rand_unif<accept_prob1);

% Sample the truncated sqrt gamma density:
u1=rand(length(x_TS1),1);
z_1=(2*delta^2./x_TS1.*gammaincinv(u1.*gammainc(z_0^2.*x_TS1./(2*delta^2),abs(lambda)),abs(lambda))).^(0.5);
% Zero out the tiny and zero ones:
z_1=z_1(~isnan(z_1));
z_1=z_1(z_1>0);

% Accept or reject:
accept_prob1a=H_0./(bessely(abs(lambda),z_1).^2+besselj(abs(lambda),z_1).^2).*(z_0.^(2*abs(lambda)-1)./z_1.^(2*abs(lambda)));
rand_unif=rand(length(z_1),1);



accept1=rand_unif<accept_prob1a;

accept_rate1a=mean(accept_prob1a)/q+accept_rate1a*(q-1)/(q);


x_GIG1=x_TS1(accept1);

exp_rnd2=exprnd(1,M,1);
gammas2=cumsum(exp_rnd2);

% Generate positive stable variable, alpha=0.5:
x2=(gammas2./(2*K2)).^(-2);

accept_prob2=exp(-x2*gamma_param^2./2).*gammainc(z_0^2*x2./(2*delta^2),0.5,'upper');%./gamma(0.5);
rand_unif=rand(length(accept_prob2),1);


accept_rate2=mean(accept_prob2)/q+accept_rate2*(q-1)/(q);


% Generate tempered stable:
x_TS2=x2(rand_unif<accept_prob2);

% Sample the truncated sqrt gamma density:
u=rand(length(x_TS2),1);
z_2=sqrt(2*delta^2./x_TS2.*gammaincinv(u.*gammainc(z_0^2.*x_TS2./(2*delta^2),0.5,'upper')+gammainc(z_0^2.*x_TS2./(2*delta^2),0.5),0.5));

% Accept or reject:
accept_prob2a=H_0./(bessely(abs(lambda),z_2).^2+besselj(abs(lambda),z_2).^2).*(z_2).^(-1);
rand_unif=rand(length(z_2),1);

accept2=rand_unif<accept_prob2a;


accept_rate2a=mean(accept_prob2a)/q+accept_rate2a*(q-1)/(q);


x_GIG2=x_TS2(accept2);

GIG_var1(q)=sum(x_GIG1);
GIG_var2(q)=sum(x_GIG2);
GIG_var(q)=GIG_var1(q)+GIG_var2(q);


GIG_points=[x_GIG1; x_GIG2];


end

if (lambda>0)
   x_GIG3=gen_gamma_process(M,gamma_param^2/2,lambda);
   GIG_var3(q)=sum(x_GIG3); 
   GIG_var(q)=GIG_var(q)+GIG_var3(q);
   
   GIG_points=[GIG_points; x_GIG3];
 
end    

N_points=length(GIG_points);
jump_times=(rand(N_points,1));

[j,ind]=sort(jump_times);
cumulative_jumps=cumsum(GIG_points(ind));

GIG_jumps_store{q}=GIG_points(ind);
Cumulative_jumps_store{q}=cumulative_jumps;
jump_times_store{q}=jump_times(ind);


end    
accept_rate1
accept_rate1a
accept_rate2
accept_rate2a

end
