import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 obj_weights=None,
                 scalarization_func=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.critic_optimizer = optim.Adam(actor_critic.base.critic.parameters(), lr=lr, eps=eps)

        self.obj_weights = None if obj_weights is None else torch.Tensor(obj_weights)

        self.scalarization_func = scalarization_func
        
    def update(self, rollouts, scalarization = None, obj_var = None):
        op_axis = tuple(range(len(rollouts.returns.shape) - 1))
        
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        if self.scalarization_func is not None or scalarization is not None:
            # recover the raw returns
            returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
            value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds

            if scalarization is not None:
                advantages = scalarization.evaluate(returns[:-1]) - scalarization.evaluate(value_preds[:-1])
            else:
                advantages = self.scalarization_func.evaluate(returns[:-1]) - self.scalarization_func.evaluate(value_preds[:-1])

        advantages = (advantages - advantages.mean(axis=op_axis)) / (
            advantages.std(axis=op_axis) + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    #print(value_pred_clipped[:,-1].mean())
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


    def constraint_update(self, rollouts, damping, d_k, max_kl, scalarization = None, obj_var = None, use_fim = True):
        op_axis = tuple(range(len(rollouts.returns.shape) - 1))

        # recover the raw returns
        returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
        value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds


        ret_1 = (returns[:-1] * torch.tensor([1.,0.])).sum(axis = -1)
        val_1 = (value_preds[:-1] * torch.tensor([1.,0.])).sum(axis = -1)
        obj_1_advantages = ret_1 - val_1
        ret_2 = (returns[:-1] * torch.tensor([0.,1.])).sum(axis = -1)
        val_2 = (value_preds[:-1] * torch.tensor([0.,1.])).sum(axis = -1)
        obj_2_advantages = ret_2 - val_2
                

        obj_1_advantages = (obj_1_advantages - obj_1_advantages.mean(axis=op_axis)) / (
            obj_1_advantages.std(axis=op_axis) + 1e-5)
        
        obj_2_advantages = (obj_2_advantages - obj_2_advantages.mean(axis=op_axis)) / (
            obj_2_advantages.std(axis=op_axis) + 1e-5)
        
        

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        #for e in range(self.ppo_epoch):
        for e in range(1):
            data_generator = rollouts.constraint_feed_forward_generator(
                    obj_1_advantages, obj_2_advantages, 1)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        cost_adv_targ, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                # update critic
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.base.critic.parameters(),
                                         self.max_grad_norm)
                self.critic_optimizer.step()
                
                # update actor
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                
                action_loss = -adv_targ * ratio
                action_loss = action_loss.mean()
                
                grads = torch.autograd.grad(action_loss, self.actor_critic.base.actor.parameters(), retain_graph=True)
                loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
                grad_norm = True
                if grad_norm == True:
                    loss_grad = loss_grad/torch.norm(loss_grad)
                #Fvp = self.Fvp_fim if use_fim else self.Fvp_direct
                Fvp = self.Fvp_fim
                stepdir = self.conjugate_gradients(Fvp, obs_batch, damping, -loss_grad, 10) #(H^-1)*g   
                if grad_norm == True:
                    stepdir = stepdir/torch.norm(stepdir)
                    
                cost_loss = -cost_adv_targ * ratio #?
                cost_loss = cost_loss.mean()
                cost_grads = torch.autograd.grad(cost_loss, self.actor_critic.base.actor.parameters(), allow_unused=True, retain_graph=True)
                cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #a
                cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad)
                cost_stepdir = self.conjugate_gradients(Fvp, obs_batch, damping, -cost_loss_grad, 10) #(H^-1)*a
                cost_stepdir = cost_stepdir/torch.norm(cost_stepdir)
                
                # Define q, r, s
                p = -cost_loss_grad.dot(stepdir) #a^T.H^-1.g
                q = -loss_grad.dot(stepdir) #g^T.H^-1.g
                r = loss_grad.dot(cost_stepdir) #g^T.H^-1.a
                s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 

                constraint_value = -value_pred_clipped[:,0]
                #print(constraint_value.mean())
                #d_k = torch.tensor(d_k).to(constraint_value.dtype).to(constraint_value.device) #???how to calculate
                d_k = constraint_value*0.99
                cc = constraint_value.mean() - d_k.mean() # c would be positive for most part of the training
                lamda = 2*max_kl
                    
                #find optimal lambda_a and lambda_b
                A = torch.sqrt((q - (r**2)/s)/(max_kl - (cc**2)/s))
                B = torch.sqrt(q/max_kl)
                if cc>0:
                    opt_lam_a = torch.max(r/cc,A)
                    opt_lam_b = torch.max(0*A,torch.min(B,r/cc))
                else: 
                    opt_lam_b = torch.max(r/cc,B)
                    opt_lam_a = torch.max(0*A,torch.min(A,r/cc))
                
                #find values of optimal lambdas 
                a = ((r**2)/s - q)/(2*opt_lam_a)
                b = opt_lam_a*((cc**2)/s - max_kl)/2
                c = - (r*cc)/s
                opt_f_a = a+b+c
    
                a = -(q/opt_lam_b + opt_lam_b*max_kl)/2
                opt_f_b = a   
                
                if opt_f_a > opt_f_b:
                    opt_lambda = opt_lam_a
                else:
                    opt_lambda = opt_lam_b
                        
                #find optimal nu
                nu = (opt_lambda*cc - r)/s
                if nu>0:
                    opt_nu = nu 
                else:
                    opt_nu = 0
                    
                """ find optimal step direction """
                # check for feasibility
                if ((cc**2)/s - max_kl) > 0 and cc>0:
                    print('INFEASIBLE !!!!')
                    #opt_stepdir = -torch.sqrt(2*max_kl/s).unsqueeze(-1)*Fvp(cost_stepdir)
                    opt_stepdir = torch.sqrt(2*max_kl/s)*Fvp(cost_stepdir, obs_batch, damping,)
                else: 
                    #opt_grad = -(loss_grad + opt_nu*cost_loss_grad)/opt_lambda
                    opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda
                    #opt_stepdir = (stepdir)/opt_lambda
                    #opt_stepdir = conjugate_gradients(Fvp, -opt_grad, 10)
                
                #print(f"{bcolors.OKBLUE} nu by lambda {opt_nu/opt_lambda},\t lambda {1/opt_lambda}{bcolors.ENDC}")
                """
                #find the maximum step length
                xhx = opt_stepdir.dot(Fvp(opt_stepdir))
                beta_1 = -cc/(cost_loss_grad.dot(opt_stepdir))
                beta_2 = torch.sqrt(max_kl / xhx)
                
                if beta_1 < beta_2:
                    beta_star = beta_1
                else: 
                    beta_star = beta_2
                
                # perform line search
                #fullstep = beta_star*opt_stepdir
                prev_params = get_flat_params_from(policy_net)
                fullstep = opt_stepdir
                expected_improve = -loss_grad.dot(fullstep)
                success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
                set_flat_params_to(policy_net, new_params)
                """
                # trying without line search
                prev_params = self.get_flat_params_from(self.actor_critic.base.actor)
                new_params = prev_params + opt_stepdir
                self.set_flat_params_to(self.actor_critic.base.actor, new_params)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    
    def conjugate_gradients(self, Avp_f, states, damping, b, nsteps, rdotr_tol=1e-10):
        x = torch.zeros(b.size(), device=b.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Avp = Avp_f(p, states, damping)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < rdotr_tol:
                break
        return x

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(self, v, states, damping):
        M, mu, info = self.actor_critic.base.actor.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set() if self.actor_critic.base.actor.is_disc_action else set([info['std_id']])

        t = torch.ones(mu.size(), requires_grad=True, device=mu.device)
        mu_t = (mu * t).sum()
        Jt = self.compute_flat_grad(mu_t, self.actor_critic.base.actor.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = self.compute_flat_grad(mu_MJv, self.actor_critic.base.actor.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not self.actor_critic.base.actor.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(self, v, states, damping):
        kl = self.actor_critic.base.actor.get_kl(states)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.actor_critic.base.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.actor_critic.base.actor.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
        
        return flat_grad_grad_kl + v * damping
    
    def compute_flat_grad(self, output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        inputs = list(inputs)
        params = []
        for i, param in enumerate(inputs):
            if i not in filter_input_ids:
                params.append(param)

        grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

        j = 0
        out_grads = []
        for i, param in enumerate(inputs):
            if i in filter_input_ids:
                out_grads.append(torch.zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
            else:
                out_grads.append(grads[j].view(-1))
                j += 1
        grads = torch.cat(out_grads)

        for param in params:
            param.grad = None
        return grads
    
    def get_flat_grad_from(self, inputs, grad_grad=False):
        grads = []
        for param in inputs:
            if grad_grad:
                grads.append(param.grad.grad.view(-1))
            else:
                if param.grad is None:
                    grads.append(torch.zeros(param.view(-1).shape))
                else:
                    grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad
    
    def set_flat_params_to(self, model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
            
    def get_flat_params_from(self, model):
        #pdb.set_trace()
        params = []
        for param in model.parameters():
            params.append(param.view(-1))

        flat_params = torch.cat(params)
        return flat_params
            