import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from math import gamma
import time

class CoalMineModel():
    def __init__(self, t1, tn, d, vartheta):
        # Initializes an instance of the model depending onthe hyperparameters
        self.d        = d
        self.vartheta = vartheta
        self.t1 = t1
        self.tn = tn
        
        
        # variables in our model get assigned after initializing them explicitly
        # which is only done within the inference procedure
        self.thetas   = None
        self.lambdas  = None
        self.ts       = None
        self.taus     = None  #Ideally I'd want to decouple the observations from the model instance itself but here we are
        self.N        = None
        self.rho      = None
        self.accepted = None
        # whenever we call np.random.gamma, we do  1/beta
        # because numpy uses "scale" format instead of "rate"
        # see the Wikipedia article on Gamma distribution
        
    def n(self, ts, taus):
        # i'th element is n_i(τ) as defined in the sheet
        # Should return an error if using outside inference function
        return np.array([np.count_nonzero((taus>ts[i]) & (taus<ts[i+1])) for i in range(self.d)])
        # TODO: Vectorize this, not scalable in d
            
        
        
    def theta_fullconditional(self, alpha,beta):
        # Used for Gibbs update of p(θ|t,λ,τ)
        # Returns a float which is a sample from the conditional distribution
        return np.random.gamma(alpha,1/beta)

    
    
    def lambda_fullconditional(self, alphas,betas):
        # Used for Gibbs update of p(λ|t,θ,τ)
        # Returns 1d numpy array of shape d which
        # is a sample from the conditional distribution
        return np.random.gamma(alphas,1.0/betas)

    
    
    def t_fullconditional(self, step, rho):
        # Used for Metropolis Hastings update of p(t|λ,θ,τ)
        # Change the argument proposal = "independent" for Independent proposal kernel
        
        dts = lambda ts: ts[1:]- ts[:-1]
        # Returns the array [t_i+1 - t_i]
        
        f   = lambda ts: 0 if (dts(ts)<0).any() else np.prod(np.exp(-self.lambdas[:,step+1]*dts(ts))*dts(ts)*(self.lambdas[:,step+1]**self.n(ts,self.taus)))   
        # Returns the posterior of t up to a normalizing constant like we defined
        # step+1 because we want to use the i+1^th lambda sample we get before sampling t
        
        new_ts       = self.ts[:,step].copy()
        # using .copy() because Python uses pass by reference i.e. if we dont copy,
        # all changes to new_ts will affect self.ts as well - which we do not want
        

        R            = rho*(self.ts[2:,step] - self.ts[:-2,step])
        epsilon      = np.random.uniform(R, -R, self.d-1)
        t_star       = self.ts[1:-1,step] + epsilon
        new_ts[1:-1] = t_star
        alpha        = min(1, f(new_ts)/f(self.ts[:,step])) # Slide 14 Lecture 9
        if np.random.rand() < alpha:
            self.accepted = self.accepted+1
            return new_ts
        else:
            return self.ts[:,step]
            
    

    def inference(self, rho, taus, N, burn_in):
        # This method performs inference on the model where rho is a tuning parameter that 
        # relates to the Metropolis Hastings update for t, taus is the observed
        # variables, N is the number of samples and burn_in is the amount of first samples
        # we discard when computing the final expectations
        assert N > burn_in
        
        M = N + burn_in
        self.rho  = rho   # Need rho and N for plots
        self.N    = N
        self.taus = taus
        
        
        # initializing θ
        self.thetas    = np.zeros(M)
        self.thetas[0] = np.random.gamma(2,1.0/self.vartheta)
    
        # initializing λ
        self.lambdas      = np.zeros((self.d,M))
        self.lambdas[:,0] = np.random.gamma(2,1.0/self.thetas[0], self.d)
        
        # initializing t
        self.ts       = np.zeros((self.d+1,M))
        self.ts[:,0]  = np.sort(np.random.uniform(self.t1, self.tn, self.d+1))
        # Making sure the years are ordered by sorting them
        self.ts[0,0]  = self.t1
        self.ts[-1,0] = self.tn
        # Assigning the fixed start and end years
        
        self.accepted = 0
        # To count the acceptance rate for Metropolis Hastings
        
        
        # Timing the sampler
        theta_time  = np.zeros(M-1)
        lambda_time = np.zeros(M-1)
        t_time      = np.zeros(M-1)
        
        
        # Starting the algorithm
        for i in range(M-1):
            # Sample from p(θ|t,λ,τ) as per the density we derived
            theta_start = time.time()
            self.thetas[i+1]    = self.theta_fullconditional(alpha=2*self.d+2, 
                                                             beta=np.sum(self.lambdas[:,i]) + self.vartheta)
            theta_time[i] = time.time() - theta_start
            
            
            # Sample from p(λ|t,θ,τ), as per the density we derived, note here we actually 
            # get a d-dimensional array where the ith component is Gamma with parameteers alphas[i],betas[i]
            lambda_start = time.time()
            self.lambdas[:,i+1] = self.lambda_fullconditional(alphas=self.n(self.ts[:,i], taus) + 2,
                                                              betas=(self.ts[1:,i]- self.ts[:-1,i])  + self.thetas[i + 1])
            lambda_time[i] = time.time() - lambda_start
            
            
            # Metropolis Hastings update for p(t|λ,θ,τ)
            # This is done according to the formulas given in the sheet, we need the current time
            # step for that which is why we pass i into the function
            t_start = time.time()
            self.ts[:,i+1]      = self.t_fullconditional(i, rho)
            t_time[i] = time.time() - t_start
        
        # Removing the samples from burn in
        self.thetas  = self.thetas[burn_in:]
        self.lambdas = self.lambdas[:,burn_in:]
        self.ts      = self.ts[:,burn_in:]
        print("Average time to sample θ is {:.7f}s, λ is {:.7f}s, t is {:.7f}s".format(np.mean(theta_time), np.mean(lambda_time), np.mean(t_time)))
        print("Metropolis-Hastings acceptance rate was {:.2f}% with d={} and rho={}".format((self.accepted/self.N)*100, self.d,self.rho))
            
            
            
    def inferred(self):
        # Method used to know if we have performed inference or not
        return (self.thetas is not None) and (self.lambdas is not None) and (self.ts is not None) and (self.N is not None) and (self.rho is not None)
        
    
        
        
    def get_posterior_samples(self):
        # Get the posterior samples of theta, lambda, t as a dict
        # Can only be done after performing inference
        if (self.inferred()):
            return {"thetas":self.thetas, "lambdas":self.lambdas, "ts":self.ts}
        else:
            print("Posteriors are None, you have not performed inference yet")
    
        
    
    def plot_posterior_histogram(self,mode="show"):
        # Plot the posterior distributions of theta, lambda, t
        # Can only be done after performing inference on the model instance
        if (self.inferred()):
            if mode=="save":
            # Plottings thetas
            
                fig_thetas,ax_thetas  = plt.subplots()
                ax_thetas.hist(self.thetas, density=True, bins=int(self.N/200))
            
                ax_thetas.set_xlabel("θ")
                ax_thetas.set_ylabel("Normed density")
                fig_thetas.savefig("theta-hist-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))

                # Plotting lambdas
                fig_lambdas, ax_lambdas = plt.subplots()
                for i in range(self.d):
                    ax_lambdas.hist(self.lambdas[i,:][self.lambdas[i,:]<8],density=True, bins=int(self.N/200), alpha = 0.75,label="λ"+str(i))
                ax_lambdas.set_xlabel("λ")
                ax_lambdas.set_ylabel("Normed density")
                ax_lambdas.legend(loc="upper right")
                fig_lambdas.savefig("lambda-hist-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))
            
                # Plottings ts
                fig_ts, ax_ts = plt.subplots()
                for i in range(1,self.d):
                    ax_ts.hist(self.ts[i,:], bins=int(self.N/2000),density=True,alpha = 0.75,label="t"+str(i+1))
                ax_ts.set_xlabel("t")
                ax_ts.set_ylabel("Normed density")
                ax_ts.legend(["t"+str(i+1) for i in range(1,self.d)],loc="upper right")
                ax_ts.set(xlim=(self.t1, self.tn))
                fig_ts.savefig("t-hist-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))
                return fig_thetas, fig_lambdas, fig_ts
            
            if mode=="show":
                main_fig,main_ax = plt.subplots(1,3,figsize=[20,4])
                main_ax[0].hist(self.thetas, density=True, bins=int(self.N/200))
            
                main_ax[0].set_xlabel("θ")
                main_ax[0].set_ylabel("Frequency density")

                # Plotting lambdas
                for i in range(self.d):
                    main_ax[1].hist(self.lambdas[i,:][self.lambdas[i,:]<8],density=True, bins=int(self.N/200), alpha = 0.75,label="λ"+str(i))
                main_ax[1].set_xlabel("λ")
                main_ax[1].set_ylabel("Normed density")
                main_ax[1].legend(loc="upper right")
            
                # Plottings ts
                for i in range(1,self.d):
                    main_ax[2].hist(self.ts[i,:], bins=int(self.N/2000),density=True,alpha = 0.75,label="t"+str(i+1))
                main_ax[2].set_xlabel("t")
                main_ax[2].set_ylabel("Normed density")
                main_ax[2].legend(loc="upper right")
                main_ax[2].set(xlim=(self.t1, self.tn))
                main_fig.suptitle("N={}-d={}-vartheta={}-rho={}".format(self.N, self.d, self.vartheta,self.rho), fontsize=16)
                return main_fig
                plt.show()
            
            
        
        else:
            print("Posteriors are None, you have not performed inference yet")
            
                
    def plot_posterior_samples(self,mode="show",freq=1):
        # Plot how samples and how the quantities of interest evolve over the MC iterations
        # Can only be done after performing inference on the model instance
        if (self.inferred()):
            colors = ["C"+str(i) for i in range(self.d)]
            if mode == "save":
            # Plot the average of theta at each iteration
                means_theta = np.cumsum(self.thetas[::freq])/(np.linspace(1,self.N,int(self.N/freq)))
                fig_thetas, ax_thetas = plt.subplots()
                ax_thetas.plot(means_theta,color=colors[0])
                ax_thetas.plot(self.thetas,alpha=0.25,color=colors[0])
                ax_thetas.set_xlabel("Monte Carlo step")
                ax_thetas.set_ylabel("Posterior mean of θ")
                fig_thetas.savefig("theta-means-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))
            
            # Plot the average of the lambdas at each iteration
                means_lambda = np.cumsum(self.lambdas[:,::freq],axis=1)/(1+np.linspace(0,self.N,int(self.N/freq)))
                fig_lambdas, ax_lambdas= plt.subplots()
                for i in range(self.d):
                    ax_lambdas.plot(means_lambda[i,:], color=colors[i],label="λ"+str(i),linewidth=2.0)
                    ax_lambdas.plot(self.lambdas[i,:], alpha=0.25,color=colors[i])
                ax_lambdas.set_xlabel("Monte Carlo step")
                ax_lambdas.set_ylabel("Posterior mean of λ")
                ax_lambdas.legend(loc="upper left")
                ax_lambdas.set(ylim=(0, 8))
                fig_lambdas.savefig("lambda-means-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))

            
            # Plot how the breakpoints change over each iteration
                fig_ts, ax_ts = plt.subplots()
                for i in range(1,self.d):
                    ax_ts.plot(self.ts[i,::freq],label="t"+str(i+1))
                ax_ts.set_xlabel("Monte Carlo step")
                ax_ts.set_ylabel("Breakpoints")
                ax_ts.set(ylim=(self.t1, self.tn))
                ax_ts.legend(loc="upper right")
                fig_ts.savefig("t-means-N{}-d{}-vartheta{}-rho{}.png".format(self.N, self.d, self.vartheta,self.rho))
                return fig_thetas, fig_lambdas, fig_ts
                                
                
            if mode == "show":
            # Plot theta samples and empirical mean at each iteration
                main_fig,main_ax = plt.subplots(1,3,figsize=[20,4])
                means_theta = np.cumsum(self.thetas[::freq])/(np.linspace(1,self.N,int(self.N/freq)))
                main_ax[0].plot(means_theta,color=colors[0])
                main_ax[0].plot(self.thetas, alpha=0.25,color=colors[0])
                main_ax[0].set_xlabel("Monte Carlo step")
                main_ax[0].set_ylabel("Posterior mean of θ")
            
            # Plot the average of the lambdas at each iteration
                means_lambda = np.cumsum(self.lambdas[:,::freq],axis=1)/(1+np.linspace(0,self.N,int(self.N/freq)))
                #main_ax[1].plot(means_lambda.T)
                for i in range(self.d):
                    main_ax[1].plot(self.lambdas[i,:],alpha=0.25,color=colors[i])
                for i in range(self.d):
                    main_ax[1].plot(means_lambda[i,:], color=colors[i],label="λ"+str(i),linewidth=2.0)
                    # Loopng twice otherwise colors arent nice
                main_ax[1].set_xlabel("Monte Carlo step")
                main_ax[1].set_ylabel("Posterior mean of λ")
                main_ax[1].set(ylim=(0, 8))
                main_ax[1].legend(loc="upper left")

            
            # Plot how the breakpoints change over each iteration
                for i in range(1,self.d):
                    main_ax[2].plot(self.ts[i,::freq],label="t"+str(i+1))
                main_ax[2].set_xlabel("Monte Carlo step")
                main_ax[2].set_ylabel("Breakpoints")
                main_ax[2].set(ylim=(self.t1, self.tn))
                main_ax[2].legend(loc="upper right")
                main_fig.suptitle("N={}-d={}-vartheta={}-rho={}".format(self.N, self.d, self.vartheta,self.rho), fontsize=16)
                return main_fig
            
                plt.show()
                
                
        else:
            print("Posteriors are None, you have not performed inference yet")