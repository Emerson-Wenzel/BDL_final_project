import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D  
from scipy.special import logit
import math

def gauss_logpdf(x, mu, s):
    normalized_x = (x - mu) / s
    if (normalized_x.detach().numpy() >= float('inf')).any():
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        print('exploded')
    logprob = (-1 * (normalized_x ** 2) / 2) - 0.5 * np.log(2 * np.pi)
    return logprob

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x - mu) * sigma.reciprocal() / math.sqrt(2)))

class BNNLayer(nn.Module):
    def __init__(self, input_dim=30, output_dim=2, prior_mu=0, prior_s=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_mu = prior_mu
        self.prior_s = prior_s
        self.usedWeights = None
        self.usedBias = None

        # Means of weights, shape input by output
        # self.W_mu_DO = nn.Parameter(torch.Tensor(input_dim, output_dim).normal_(prior_mu, prior_s))

        self.W_mu_DO = nn.Parameter(torch.Tensor(input_dim, output_dim).normal_(0, 0.1))
        # Log std of weights, shape input by output
        self.W_log_s_DO = nn.Parameter(torch.Tensor(input_dim, output_dim).normal_(0, 0.1))

        # Means of biases, shape output dim
        self.b_mu_O = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.1))
        # Log stds of biases, shape output dim
        self.b_log_s_O = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.1))

        self.log_prior = 0
        self.log_post_est = 0
        

    def W_b_by_reparam(self):      
        # Reparameterization
        rand_norm_DO = torch.Tensor(self.input_dim, self.output_dim).normal_(0, 1)
        rand_norm_O = torch.Tensor(self.output_dim).normal_(0, 1)
        W_DO = self.W_mu_DO + rand_norm_DO * torch.exp(self.W_log_s_DO)
        b_O = self.b_mu_O + rand_norm_O * torch.exp(self.b_log_s_O)
        return (W_DO, b_O)



    # forward for one layer will sample the weights for that layer, and output 
    # the full output of that layer!

    # Return matrix [num_preds x output_size]
    # if predict=False (i.e. being used in training), then we reuse same posterior 
    # draw for each prediction. If predict = True, then we use different draws from the
    # posterior for each vectorized prediction

    # D is input dimension, O is output dimension 
    def forward(self, X_ND, predict=False, num_preds=1):
        # IMPLEMENT IF PREDICT, REUSE SAME POSTERIOR DRAW IN PREDICTION

        if predict:
            # print("W_mu_DO shape: ", self.W_mu_DO.shape)
            (W_DO, b_O) = self.W_b_by_reparam()
            self.usedWeights = W_DO
            self.usedBias = b_O
#             pred = torch.mm(X_ND, W_DO) + b_O.expand(X_ND.size()[0], self.output_dim)
            pred = torch.mm(X_ND, W_DO) + b_O
            return pred

        (W_DO, b_O) = self.W_b_by_reparam()

        # Note: Updating batch by batch, not datapoint by datapoint
        # OK, more than one prediction at once getting very confusing. Leaving as 
        # is for now, come back to it later to decide how good of an idea it actually is
        if num_preds > 1:
            outputs = []
            log_priors = []
            log_post_ests = []
            for i in range(num_preds):
                output = torch.mm(X_ND, W_DO) + b_O.expand(X_ND.size()[0], self.output_dim)
                outputs.append(output.reshape(1, output.shape[0], output.shape[1]))
                log_prior = gauss_logpdf(W_DO, self.prior_mu, self.prior_s).sum() + gauss_logpdf(b_O, self.prior_mu, self.prior_s).sum()
                log_priors.append(log_prior.reshape(1, 1))
                log_post_est = (gauss_logpdf(W_DO, self.W_mu_DO, torch.exp(self.W_log_s_DO)).sum() +
                                gauss_logpdf(b_O, self.b_mu_O, torch.exp(self.b_log_s_O)).sum())
                log_post_ests.append(log_post_est.reshape(1, 1))
            output = torch.cat(outputs, dim=0)
            self.log_prior = torch.cat(log_priors, dim=0)
            self.log_post_ests = torch.cat(log_post_ests, dim=0)            

        elif num_preds == 1:
            output = torch.mm(X_ND, W_DO) + b_O
            # print("output shape: ", output.shape)
            self.log_prior = (gauss_logpdf(W_DO, self.prior_mu, self.prior_s).sum() + 
                              gauss_logpdf(b_O, self.prior_mu, self.prior_s).sum()) 
            

            self.log_post_est = (gauss_logpdf(W_DO, self.W_mu_DO, torch.exp(self.W_log_s_DO)).sum() +
                                 gauss_logpdf(b_O, self.b_mu_O, torch.exp(self.b_log_s_O)).sum())
        else:
            raise(ValueError)
        return output


class BNN(nn.Module):
    def __init__(self, input_dim, core_hidden_layers=[], mu_hidden_layers=[], log_s_hidden_layers=[], prior_mu=0, prior_s=0.01,
                 classification=False):
          super().__init__() 
          self.input_dim = input_dim
          self.prior_mu = prior_mu
          self.prior_s = prior_s
#           self.activ = nn.LeakyReLU()
          self.activ = nn.Tanh()
          self.core_layers = nn.ModuleList()
          self.mu_layers = nn.ModuleList()
          self.log_s_layers = nn.ModuleList()

          core_hidden_layers = []
          mu_hidden_layers = []
          log_s_hidden_layers = []
          core_layer_list = [input_dim] + core_hidden_layers

          # if core_layer_list only input_dim, then only build mu and log_s layers (i.e. no shared layers)
          if len(core_layer_list) > 1:
              # build core layers
              for i, layer_size_i in enumerate(core_layer_list):
                  if i == 0:
                      continue
                  else:
                      self.core_layers.append(BNNLayer(core_layer_list[i - 1], layer_size_i, self.prior_mu, self.prior_s))
                  
          # both layer lists below will contain sizes for hidden layers and always end with 1 for the size of output layer
          mu_layer_list = [core_layer_list[-1]] + mu_hidden_layers + [1]
          log_s_layer_list = [core_layer_list[-1]] + log_s_hidden_layers + [1]
 
          # build mu layers after split head
          for i , layer_size_i in enumerate(mu_layer_list): 
              if i == 0:
                  continue
              else:
                  self.mu_layers.append(BNNLayer(mu_layer_list[i - 1], layer_size_i, self.prior_mu, self.prior_s))

          # build log_s layers after split head
          for i , layer_size_i in enumerate(log_s_layer_list): 
              if i == 0:
                  continue
              else:
                  self.log_s_layers.append(BNNLayer(log_s_layer_list[i - 1], layer_size_i, self.prior_mu, self.prior_s))

          self.classification = classification

          
          # Not used for std dev prediction
          self.classification_threshold = 0.5
          self.pred_sigmoid = nn.Sigmoid()


    def forward(self, X_ND, predict=False, num_preds=1, threshold=True):
          if isinstance(X_ND, np.ndarray):
              X_ND = torch.tensor(X_ND, dtype=torch.float)


          output = X_ND
          if len(self.core_layers) > 0:
              for layer in self.core_layers:
                  output = self.activ(layer(output, predict, num_preds))

          # Feel like this might be wrong and we might need to do some 
          # sort of deep copy here

          mu_output = output
          log_s_output = output

#           print('mu_output: ', mu_output)
          for i, layer in enumerate(self.mu_layers):
              if i != len(self.log_s_layers) - 1:
                  mu_output = self.activ(layer(mu_output, predict, num_preds))
              else:
                  # output layer -- no activation
                  mean = layer(mu_output, predict, num_preds)

#           print('log_s_output: ', log_s_output)
          for i, layer in enumerate(self.log_s_layers):
              if i != len(self.log_s_layers) - 1:
                  log_s_output = self.activ(layer(log_s_output, predict, num_preds))
              else:
                  # output layer -- no activation
                  log_s = layer(log_s_output, predict, num_preds)


          output = torch.stack((mean, log_s), dim=1).reshape(-1, 2)
          
          if predict:

              if self.classification:
                  continuous_pred = output[:,0]
                  # print("continuous pred: ", continuous_pred[:5].detach().numpy())

                  # need to look if 1 is churn or not churn in dataset
                  prob_of_one = self.pred_sigmoid(continuous_pred)
                  if threshold:
                      pred = (prob_of_one > self.classification_threshold)
                      output = pred
                  else:
                      output = prob_of_one
              else:
                  output = output[:,0]

          return output

    def calc_total_log_prior_log_post_est(self):
          if len(self.core_layers) > 0:
              total_log_prior_core = sum([layer.log_prior for layer in self.core_layers])
          else:
              total_log_prior_core = 0
          total_log_prior_mu = sum([layer.log_prior for layer in self.mu_layers])
          total_log_prior_log_s = sum([layer.log_prior for layer in self.log_s_layers])
          total_log_prior = total_log_prior_core + total_log_prior_mu + total_log_prior_log_s 

          if len(self.core_layers) > 0:
              total_log_post_est_core = sum([layer.log_post_est for layer in self.core_layers])
          else:
              total_log_post_est_core = 0
          total_log_post_est_mu = sum([layer.log_post_est for layer in self.mu_layers]) 
          total_log_post_est_log_s = sum([layer.log_post_est for layer in self.log_s_layers])
          total_log_post_est = total_log_post_est_core + total_log_post_est_mu + total_log_post_est_log_s 

          return total_log_prior, total_log_post_est


class BNNBayesbyBackprop(nn.Module):
    # core_hidden_layers is list of hidden sizes for all shared layers of the network
    # mu_hidden_layers is list of hidden sizes for all hidden layers specific to the mu head
    # log_s_hidden_layers is list of hidden sizes for all hidden layers specific to the log_s head
    def __init__(self, input_dim=2, core_hidden_layers=[], mu_hidden_layers=[], log_s_hidden_layers=[], 
                 prior_mu=10, prior_s=0.05, num_MC_samples=100, classification=False, homoscedastic_var=None):
        super().__init__()
        
        self.prior_mu = prior_mu
        self.prior_s = prior_s
        self.num_MC_samples = num_MC_samples
        self.classification = classification
        if homoscedastic_var:
            self.homoscedastic_std = np.sqrt(homoscedastic_var)
        else:
            self.homoscedastic_std = None
        self.class_weights = {0: 1, 1: 1}
        self.mean_likelihood = None
        self.log_prior = None
        self.log_posterior = None 
        self.elbo = None 
        self.gradB = None
        self.reg = None

        # self.model = BNN(38, self.prior_mu, self.prior_s)
        self.model = BNN(input_dim, core_hidden_layers, mu_hidden_layers, log_s_hidden_layers,
                         self.prior_mu, self.prior_s, classification)

#         for debugging
        self.last_batch = False


    # @TODO: Does this scale with "batch size" or "traning set size" or what?
    def MC_elbo(self, X_ND, y_N, curr_batch, n_batches, epoch):
#         self.model.zero_grad()        
        # out[0] is the predicted mean, out[1] is the predicted std_dev
        aggregate_log_prior, aggregate_log_post_est, aggregate_log_likeli, aggregate_log_s_N = 0.0, 0.0, 0.0, 0.0
        for i in range(self.num_MC_samples):
            nn_output_Nx2 = self.model(X_ND)
            nn_output_mu_N = nn_output_Nx2[:,0]
            nn_output_log_s_N = nn_output_Nx2[:,1]
            if epoch < 60:
#                 nn_output_log_s_N = torch.clamp(nn_output_log_s_N, min=np.log(0.1), max=np.log(0.1))
                nn_output_log_s_N = torch.clamp(nn_output_log_s_N, min=np.log(0.001), max=np.log(10))
#             elif (epoch >= 50) and (epoch < 70):
#                 nn_output_log_s_N = torch.clamp(nn_output_log_s_N, min=np.log(0.01), max=np.log(5))
#             elif (epoch >= 15) and (epoch < 30):
#                 nn_output_log_s_N = torch.clamp(nn_output_log_s_N, min=np.log(0.0001), max=25)

            # Aggregated probabilities across an entire batch/trainset for each sample
            sample_log_likeli = self.likelihood_est(y_N, nn_output_mu_N, nn_output_log_s_N)
            sample_log_prior, sample_log_post_est = self.model.calc_total_log_prior_log_post_est()

            # Aggregating probabilities across all samples
            aggregate_log_prior += sample_log_prior
            aggregate_log_post_est += sample_log_post_est
            aggregate_log_likeli += sample_log_likeli
            aggregate_log_s_N += nn_output_log_s_N.sum()
#                 print('log_s_N max\t', nn_output_log_s_N.detach().numpy().max())
#                 print('log_s_N min\t', nn_output_log_s_N.detach().numpy().min())

        scalar = 1

        self.log_prior = aggregate_log_prior.detach().numpy() / self.num_MC_samples
        self.log_posterior = aggregate_log_post_est.detach().numpy() / self.num_MC_samples
        self.mean_likelihood = aggregate_log_likeli.detach().numpy() / self.num_MC_samples
        self.reg = scalar * (aggregate_log_s_N.detach().numpy() / self.num_MC_samples)

#        if epoch >= 40:
#            print("{} {} {}".format(self.log_prior, self.log_posterior, self.mean_likelihood))

#       *************************** The below code block is necessary to manually test the gradient calculation *****************************
#
#         if curr_batch == (n_batches - 1):
          #We assume that it is a scalar representing the total log prior(w, b) across all samples

#             self.mean_likelihood = aggregate_log_likeli.detach().numpy() / self.num_MC_samples
#             self.elbo = (-1 * (aggregate_log_prior + aggregate_log_likeli - aggregate_log_post_est) / self.num_MC_samples)  
#             self.elbo.backward()
# #             print("\ngrads w1 ", self.model.l1.W_mu_DO.grad[:,0])
# #             self.gradB = self.model.l1.b_mu_O.grad[0] 
# #             print("grad b ", self.gradB)
#             print("\ngrads w1 ", self.model.mu_layers[0].W_mu_DO.grad[:,0])
#             self.gradB = self.model.mu_layers[0].b_mu_O.grad[0] 
#             print("grad b ", self.gradB)

#       **************************************************************************************************************************************

#         Old loss function which had the bias term which seemed to help with convergence for regression
#         loss = (-1 * (aggregate_log_prior + aggregate_log_likeli - aggregate_log_post_est) + (aggregate_log_s_N * scalar)) / self.num_MC_samples

        loss = (-1 * (aggregate_log_prior + aggregate_log_likeli - aggregate_log_post_est)) / self.num_MC_samples
#         print('log_prior: ', self.log_prior, '\tlog_posterior:', self.log_posterior,
#                 '\tlikelihood: ', self.mean_likelihood, '\treg: ', self.reg)

        return loss 

    # @TODO: is it gauss_logpdf(y, sigmoid(pred_y), exp(nn_ouput_log_s_N))? or is it: MC sample: sigmoid(sample from N(pred_y, exp(nn_ouput_log_s_N))), threshold at [0.5]?
    def likelihood_est(self, y_N, nn_output_mu_N, nn_output_log_s_N, MC_samples=20):

        if self.classification:
            sigmoid = nn.Sigmoid()
            s_N_list = []

            for i in range(MC_samples): 
                e = torch.Tensor(size=(y_N.shape)).normal_(0, 1.0)
                if self.homoscedastic_std:
                    stds = self.homoscedastic_std
                else:
                    stds = torch.exp(nn_output_log_s_N)
                s_N = nn_output_mu_N + e * stds
                s_N_list.append(s_N)

            # s_NxMC is NxMC where each row holds MC samples, s ~ N(mu(X_n), log_s(X_n))
            # where mu(X_n) and log_s(X_n) are the BNN's predicted values for nth instances
            s_NxMC = torch.stack(s_N_list, dim=1)

            probs_of_one_NxMC = sigmoid(s_NxMC)
            avg_prob_of_one_N = probs_of_one_NxMC.mean(dim=1)
            likelihood_N = torch.empty(size=y_N.shape)

            # fill likelihood_N with probability of 0 or 1 depending on true label
            likelihood_N[y_N == 0] = 1 - avg_prob_of_one_N[y_N == 0]
            likelihood_N[y_N == 1] = avg_prob_of_one_N[y_N == 1]

            # add small number in log for numerical stability
            log_likelihood_N = torch.log(likelihood_N + 1e-7)  
            log_likelihood = log_likelihood_N.sum()

        else:
            stds = torch.exp(nn_output_log_s_N)
            log_likelihood_N = gauss_logpdf(y_N.reshape(-1), nn_output_mu_N, stds)
            # log_likelihood_N = gauss_logpdf(y_N.reshape(-1), nn_output_mu_N, 0.01)
            log_likelihood = log_likelihood_N.sum()

        return log_likelihood


    def _logging_header(self):
        core_layer_list = []
        for i, layer_i in enumerate(self.model.core_layers):
            # layer.output_dim is the number of hidden nodes
            for n in range(layer_i.input_dim):
                for o in range(layer_i.output_dim):
                    core_layer_list.append("l{}_w{}{}".format(i, n, o))

        mu_layer_list = []
        for i, layer_i in enumerate(self.model.mu_layers):
            # layer.output_dim is the number of hidden nodes
            for n in range(layer_i.input_dim):
                for o in range(layer_i.output_dim):
                    mu_layer_list.append("mu_l{}_w{}{}".format(i, n, o))

        log_s_layer_list = []
        for i, layer_i in enumerate(self.model.log_s_layers):
            # layer.output_dim is the number of hidden nodes
            for n in range(layer_i.input_dim):
                for o in range(layer_i.output_dim):
                    log_s_layer_list.append("log_s_l{}_w{}{}".format(i, n, o))

#         print(core_layer_list + mu_layer_list + log_s_layer_list)
        strToWrite = ','.join(map(str, core_layer_list + mu_layer_list + log_s_layer_list))
        header = strToWrite + ",log_prior,log_posterior,mean_likelihood,reg\n"
        return header

    def fit(self, X, y, learning_rate=0.001, n_epochs=100, batch_size=1000, plot=False, weight_classes=False, verbose=False, logging=False):
        if logging:
            loggingFileName = str(int(time.time())) + ".csv"
            loggingFileName = "logging.csv"
            if verbose:
                print("Data being saved in following file:\n{}".format(loggingFileName))
            logger = open(loggingFileName, "w")
            header = self._logging_header()
            logger.write(header)
            logger.close()

        if weight_classes:
            n_samples = y.shape[0]
            n_classes = 2
            class_weights = n_samples / (n_classes * np.bincount(y))
            class_weights = class_weights / 30
            print('class_weights:', class_weights)
            self.class_weights[0] = class_weights[0]
            self.class_weights[1] = class_weights[1]
        
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        X_batch = torch.Tensor()
        y_batch = torch.Tensor()
        loss_by_epoch = []
        e = 0
        for e in range(n_epochs):
            batch_losses = []
            for batch_num in range(n_batches):
                if batch_num == (n_batches - 1):
                    self.last_batch = True
                else:
                    self.last_batch = False
                batch_start_i = batch_size * batch_num
                if (batch_size * (batch_num + 1)) < X.shape[0]:
                    batch_end_i = batch_size * (batch_num + 1)
                else:
                    batch_end_i = X.shape[0]

                self.model.zero_grad()
                X_batch = torch.Tensor(X[batch_start_i : batch_end_i])
                y_batch = torch.Tensor(y[batch_start_i : batch_end_i])
                loss = self.MC_elbo(X_batch, y_batch, batch_num, n_batches, e)
                batch_losses.append(loss.detach().numpy())
                loss.backward()
                optimizer.step()

            if logging:
                toWrite = []
                # get weights
                for layer in self.model.core_layers:
                    toWrite.append(layer.W_mu_DO.detach().numpy().T.flatten())
                for layer in self.model.mu_layers:
                    toWrite.append(layer.W_mu_DO.detach().numpy().T.flatten())
                for layer in self.model.log_s_layers:
                    toWrite.append(layer.W_mu_DO.detach().numpy().T.flatten())
                # weight grads
                # biases
                # bias grads
                toWrite = [item for sublist in toWrite for item in sublist] + [self.log_prior, self.log_posterior, self.mean_likelihood, self.reg]
    #             toWrite = [item for sublist in toWrite for item in sublist] + [self.log_prior, self.log_posterior, self.mean_likelihood, self.reg]
                # w1_1, w1_2, w2_1, w2_2, w1_1_grad, w1_2_grad, w2_1_grad, w2_2_grad, b_1, b_2, b_1_grad, b_2_grad, log_prior, log_posterior, mean_likelihood
                strToWrite = ','.join(map(str, toWrite))
                logger = open(loggingFileName, "a")
                logger.write(strToWrite + '\n')
                logger.close()

##                if np.isnan(self.model.l1.W_mu_DO.detach().numpy()).any():
#                    breakTime = True
#                    break
#            if breakTime == True:
#                break
#             print("full weights: \n", self.model.l1.W_mu_DO.detach().numpy())
#             b1 = self.model.l1.b_mu_O.detach().numpy()[0]
#             w1 = self.model.l1.W_mu_DO.detach().numpy()[0]
#             x1 = np.random.uniform(-8, 8, (100,2))
#             w1 = w1.reshape((-1, 1))
#             print(w1.shape)
#             y1 = x1 @ w1 + b1
            
            X_batch_np = X_batch.detach().numpy() 
            y_batch_np = y_batch.detach().numpy()
#             plt.scatter(X_batch_np[y_batch_np == 0, 0], X_batch_np[y_batch_np == 0, 1], c='red', alpha=.2)
#             plt.scatter(X_batch_np[y_batch_np == 1, 0], X_batch_np[y_batch_np == 1, 1], c='blue', alpha=.2)
#             plt.show()
#             fig = plt.figure(figsize=(20,20))
#             ax = fig.add_subplot(311, projection='3d')
#             print("------")
#             print(x1[:,0].shape)
#             print(x1[:,1].shape)
#             print(y1.shape)
#             print("------")
#             ax.scatter(x1[0,1], x1[:,1], y1)
#             ax.set_xlabel('X_train[:,0]')
#             ax.set_ylabel('X_train[:,1]')
#             ax.set_zlabel('output[:,1] (Standard Deviation)')
#             fig.show()

            output = self.model(X_batch)

            X_full = torch.Tensor(X)
            y_full = torch.Tensor(y)



            pred = self.model(X_full, predict=True)
#            print('used weights1: ',self.model.l1.usedWeights[:,0].detach().numpy())
#            print('used bias1: ', self.model.l1.usedBias[0].detach().numpy())
 
#            pred2 = self.model(X_full, predict=True)
#            print('used weights2: ',self.model.l1.usedWeights[:,0].detach().numpy())
#            print('used bias2: ', self.model.l1.usedBias[0].detach().numpy())
#            differences = (pred.detach().numpy() == pred2.detach().numpy()).astype(int).sum() / pred.shape[0]

#            print("differences between preds: {}".format(differences))
#            print("standard deviation: {}".format(torch.exp(self.model.l1.W_log_s_DO)[:,0].detach().numpy()))
#            print("bias std:           {}".format(torch.exp(self.model.l1.b_log_s_O)[0].detach().numpy()))
            #pred = self.model(X_full, predict=True)
            cur_epoch_loss = np.array(batch_losses).sum()

            # classification accuracy
            if self.classification:
                pred_np = pred.detach().numpy() 
                y_full_np = y_full.numpy() 
                acc = (pred_np == y_full_np).astype(int).sum() / y_full.shape[0]

                true_pos = pred_np[y_full_np == 1].sum()
                total_real_pos = y_full_np[y_full_np == 1].shape[0]
                pred_pos = pred_np[pred_np == 1].shape[0]
                try:
                    precision = true_pos / pred_pos 
                except ZeroDivisionError: 
                    precision = 0

                recall = true_pos / total_real_pos  
#                 print('var weight: ', self.model.l1.W_mu_DO[0][1].detach().numpy(), 'bias: ', self.model.l1.b_mu_O[1].detach().numpy())
                if verbose:
                    print("Epoch: ", e, "\tLoss: ", cur_epoch_loss, "\tacc: ", acc)
#                 print("Epoch: ", e, "\tLoss: ", cur_epoch_loss, "\tacc: ", acc,
#                       '\tb: ', self.model.l1.b_mu_O[1].detach().numpy(), '\tW_1:', self.model.l1.W_mu_DO[0][1].detach().numpy(),
#                       '\tW_2: ',  self.model.l1.W_mu_DO[1][1].detach().numpy())
#                 print("Epoch: ", e, "\tLoss: ", cur_epoch_loss, "\tacc: ", acc, '\tprec: ', precision, '\trec: ', recall)
            else: 
            # regression accuracy
                MAE = torch.abs(pred - y_full.flatten()).mean().detach().numpy()
                if verbose:
                    print("Epoch: ", e, "\tLoss: ", cur_epoch_loss, "\tMAE: ", MAE, 
                          '\tb: ', self.model.l1.b_mu_O[1].detach().numpy(), '\tW:', self.model.l1.W_mu_DO[1][1].detach().numpy())
            # print()
            # print("pred: ", pred.detach().numpy()[:5])
            # print("real: ", y_full.numpy().reshape(-1)[:5])
            loss_by_epoch.append(cur_epoch_loss)
            # e += 1
        
        if plot:
            plt.plot([i for i in range(n_epochs)], loss_by_epoch)



