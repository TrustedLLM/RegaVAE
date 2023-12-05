import torch
import os
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import logging
from tqdm import tqdm
from train_utils import *
import pandas as pd
from autofaiss import build_index
logger = logging.getLogger(__name__)

def create_mask(num_tokens, max_len):
        base_position_matrix = torch.arange(
            0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
        mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
        return mask
def collate_wiki(text,tokenizer,args):
    source = [s + [tokenizer.eos_id] for s in text]
    source = [item[:680] for item in source]
    source_length_list = [len(s) for s in source]
    source_max_t = max(source_length_list)
    new_source = [s + [tokenizer.pad_id] * (source_max_t - len(s)) for s in source]
    new_source = torch.LongTensor(new_source)
    source_attention_mask = create_mask(torch.LongTensor(source_length_list), source_max_t)
    return new_source.to(args.device),source_attention_mask.to(args.device)

def range_batch(max_value, batch_size): 
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)  
        counter = curr
def ctext_to_embeddings(df,model,tokenizer,args):
    model.eval()
    texts = df['text'].values
    latents =[]
    texts_input=[]
    for text in tqdm(texts, desc="prepare input_ids"):
        text_input = tokenizer.encode(text)
        texts_input.append(text_input)
    for dim_slice in tqdm(range_batch(len(texts),8),desc="prepare embeddings"):
        text_input = texts_input[dim_slice]
        text_input_ids,text_input_masks = collate_wiki(text_input,tokenizer,args)
        text_input_ids = text_input_ids.cuda()
        text_input_masks = text_input_masks.cuda()
        with torch.no_grad(): 
            output = model.get_prior(len(text_input_ids),torch.device('cuda'),condition=text_input_ids,condition_mask = text_input_masks)

        tmp_latent = np.stack([i.cpu().numpy() for i in output],axis=0)
        
        for j in range(len(output[0])):
            latents.append(tmp_latent[:,j])
            
        
    df['latents'] = latents
    
    return df
def text_to_embeddings(df,model,tokenizer,args):
    model.eval()
    
    texts = df['text'].values
    latents =[]
    all_post_mu = []
    all_post_sigma=[]
    texts_input=[]
    for text in tqdm(texts, desc="prepare input_ids"):
        text_input = tokenizer.encode(text)
        texts_input.append(text_input)
    for dim_slice in tqdm(range_batch(len(texts),8),desc="prepare embeddings"):
        text_input = texts_input[dim_slice]
        text_input_ids,text_input_masks = collate_wiki(text_input,tokenizer,args)
        with torch.no_grad():    
            output = model(text_input_ids,attention_mask = text_input_masks)
        latent=[]
        post_mu =[]
        post_sigma=[]
        tmp_latent = np.stack([i.cpu().numpy() for i in output.latent],axis=0)
        tmp_post_mu = np.stack([i.cpu().numpy() for i in output.all_post_mu],axis=0)
        tmp_post_sigma = np.stack([i.cpu().numpy() for i in output.all_post_sigma],axis=0)
        # print(tmp_latent.shape)
        for j in range(len(output.latent[0])):
            # print(j)
            latent.append(tmp_latent[:,j])
            post_mu.append(tmp_post_mu[:,j])
            post_sigma.append(tmp_post_sigma[:,j])
        latents.extend(latent)
        all_post_mu.extend(post_mu)
        all_post_sigma.extend(post_sigma)
        
    df['latents'] = latents
    df['all_post_mu'] = all_post_mu
    df['all_post_sigma'] = all_post_sigma
    
    return df

def text_to_embeddings_bm25(texts,model,tokenizer,args):
    model.eval()
    
    latents =[]
    all_post_mu = []
    all_post_sigma=[]
    texts_input=[]
    for text in texts:
        text_input = tokenizer.encode(text)
        texts_input.append(text_input)
    for dim_slice in range_batch(len(texts),8):
        text_input = texts_input[dim_slice]
        text_input_ids,text_input_masks = collate_wiki(text_input,tokenizer,args)
        with torch.no_grad():    
            output = model(text_input_ids,attention_mask = text_input_masks)
        latent=[]
        post_mu =[]
        post_sigma=[]
        tmp_latent = np.stack([i.cpu().numpy() for i in output.latent],axis=0)
        tmp_post_mu = np.stack([i.cpu().numpy() for i in output.all_post_mu],axis=0)
        tmp_post_sigma = np.stack([i.cpu().numpy() for i in output.all_post_sigma],axis=0)
        for j in range(len(output.latent[0])):
            # print(j)
            latent.append(tmp_latent[:,j])
            post_mu.append(tmp_post_mu[:,j])
            post_sigma.append(tmp_post_sigma[:,j])
        
        latents.extend(latent)
        all_post_mu.extend(post_mu)
        all_post_sigma.extend(post_sigma)
        
    df['latents'] = latents
    df['all_post_mu'] = all_post_mu
    df['all_post_sigma'] = all_post_sigma
    
    return df

def index_embeddings(  
    embeddings
):
    index, _ = build_index(embeddings, save_on_disk=False,verbose=40)
    
    return index
def prepare_for_training(args, model, train_iter):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=True)
    t_total = len(train_iter) * args.epochs
    if args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = None

    return model, optimizer, scheduler

def compute_loss(logits, target_tokens, kl_loss=None, beta=None, ignore_index=50256):
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_tokens[..., 1:].contiguous()
    
    ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    if kl_loss is not None:
        loss = ce_loss + beta * kl_loss
    else:
        loss = ce_loss
    return loss, ce_loss, kl_loss

def train(model, train_iter, valid_iter, args, df,df_valid, tokenizer,LOGGER):
    logging.info('begin trainging...')
    model, optimizer, scheduler = prepare_for_training(args, model, train_iter)
    if args.cycle_annealing:
        beta = 1e-5
        beta_0 = 1e-5
    else:
        beta = 1
    global_step = 0
    
    one_epoch_step = len(train_iter) // args.gradient_accumulation_steps
    beta_zero = beta_increase = args.cycle_iters // 2
    running_loss = 0
    running_ce_loss = 0
    running_kl_loss = 0
    running_bow_loss = 0
    for epoch in range(args.epochs):
        if args.dataset_type == 'vae':
            df = text_to_embeddings(df,model.encoder,tokenizer,args)
        else:
            df = ctext_to_embeddings(df,model.encoder,tokenizer,args)
        index_list = []
        all_add_latents = df['latents'].values  
        
        if epoch < args.bm25_epoch:
            index_list = None
        else:
            for i in range(12):
                tmp_index = index_embeddings(df['latents'].values[:][i])
                index_list.append(tmp_index)
        model.train()
        for i, inputs in enumerate(train_iter):
            if epoch < args.bm25_epoch:
                temp_latents=[]
                temp_scores=[]
                for text in inputs['texts']:
                    result_index = df[df.text == text]['result_indexs'].values[0]
                    result_score = df[df.text == text]['result_scores'].values[0]
                    temp_latents.append([all_add_latents[item] for item in result_index[:args.neighbors]]) #bs,k,num_layer,latent_size
                    temp_scores.append(result_score[:args.neighbors])#k bs:k
                    
                add_latent = torch.from_numpy(np.array(temp_latents)).to(args.device)
                similarity = torch.from_numpy(np.array(temp_scores,dtype =np.float32)).to(args.device)
                add_latent = add_latent.permute(2,0,1,3)
                similarity = similarity.unsqueeze(0)
                similarity = similarity.expand(12,similarity.shape[1],similarity.shape[2])
                similarity = torch.softmax(similarity,dim=-1)
            else:
                
                with torch.no_grad():
                    query_output = model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                add_latent=[]
                similarity=[]
                tmp_latent = np.stack([i.cpu().numpy() for i in query_output.latent],axis=0)
                for v in range(12):
                    query_vector=tmp_latent[v]
                    distances, indices = index_list[v].search(query_vector, k = args.neighbors) #bs:k bs:k
                    similarity.append(distances)#nl:bs:k
                    temp_bs_latent=[]
                    for g in range(len(indices)):
                        temp_latent = []
                        for k in range(args.neighbors):
                            temp_latent.append(all_add_latents[indices[g,k]][v]) #k,latent_size
                        temp_bs_latent.append(temp_latent)  #bs:k:latent_size
                    add_latent.append(temp_bs_latent)#num_layer:bs:k:lz
                add_latent = torch.from_numpy(np.array(add_latent)).to(args.device)
                similarity = torch.from_numpy(np.array(similarity)).to(args.device)
                similarity = torch.softmax(similarity,dim=-1)
            del inputs['texts']
            inputs['add_latent']=add_latent
            inputs['similarity'] = similarity
            model_output = model(**inputs)
            if args.use_bow:
                ce_loss, kl_loss, bow_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss + args.bow_weight * bow_loss
            else:
                ce_loss, kl_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss
                    
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss = loss.mean()
            loss.backward()
            
            running_loss += loss.item()
            running_ce_loss += ce_loss.mean().item() / args.gradient_accumulation_steps
            running_kl_loss += kl_loss.mean().item() / args.gradient_accumulation_steps
            if args.use_bow:
                running_bow_loss += bow_loss.mean().item() / args.gradient_accumulation_steps
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if args.cycle_annealing:
                    one_period = epoch % args.cycle_iters
                    if one_period < beta_zero:
                        beta = beta_0
                    else:
                        beta = min(1.0, beta + (1 - beta_0) / (beta_increase * one_epoch_step / 2))
                
                if epoch >= args.bm25_epoch and global_step % args.rebuild_index_step  == 0:
                    if args.dataset_type == 'vae':
                        df = text_to_embeddings(df,model.encoder,tokenizer,args)
                    else:
                        df = ctext_to_embeddings(df,model.encoder,tokenizer,args)
                    index_list = []
                    all_add_latents = df['latents'].values
                    for num_layer in range(12):
                        tmp_index = index_embeddings(df['latents'].values[:][num_layer])
                        index_list.append(tmp_index)

                if global_step % args.log_step == 0:
                    LOGGER.info('training loss: step [{}~{}], loss {}, ce_loss {}, kl_loss {}, bow_loss {}, lr {}, beta {}'.
                        format(global_step - args.log_step, global_step, running_loss / args.log_step, running_ce_loss / args.log_step, 
                                running_kl_loss / args.log_step, running_bow_loss / args.log_step, optimizer.param_groups[0]['lr'], beta))
                    running_loss = 0
                    running_kl_loss = 0
                    running_ce_loss = 0
                    running_bow_loss = 0

        valid(model, valid_iter, epoch, args,df,df_valid,index_list,all_add_latents,LOGGER)
        save(model, args, epoch)
    LOGGER.info('training finished')

def valid(model, valid_iter, epoch, args,df,df_valid,index_list,all_add_latents,LOGGER,beta=1):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        valid_kl_loss = 0
        valid_ce_loss = 0
        valid_bow_loss = 0
        for inputs in tqdm(valid_iter, desc='valid epoch {}'.format(epoch)):
            if epoch < args.bm25_epoch:
                temp_latents=[]
                temp_scores=[]
                for text in inputs['texts']:
                    result_index = df_valid[df_valid.text == text]['result_indexs'].values[0]
                    result_score = df_valid[df_valid.text == text]['result_scores'].values[0]
                    temp_latents.append([all_add_latents[item] for item in result_index[:args.neighbors]]) #bs,k,num_layer,latent_size
                    temp_scores.append(result_score[:args.neighbors])#k bs:k
                    
                add_latent = torch.from_numpy(np.array(temp_latents)).to(args.device)
                similarity = torch.from_numpy(np.array(temp_scores,dtype =np.float32)).to(args.device)
                add_latent = add_latent.permute(2,0,1,3)
                similarity = similarity.unsqueeze(0)
                similarity = similarity.expand(12,similarity.shape[1],similarity.shape[2])
                similarity = torch.softmax(similarity,dim=-1)
            else:
                with torch.no_grad():
                
                    query_output = model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                add_latent=[]
                similarity=[]
                tmp_latent = np.stack([i.cpu().numpy() for i in query_output.latent],axis=0)
                for v in range(12):
                    query_vector=tmp_latent[v]
                    distances, indices = index_list[v].search(query_vector, k = args.neighbors) #bs:k bs:k
                    similarity.append(distances)#nl:bs:k
                    temp_bs_latent=[]
                    for g in range(len(indices)):
                        temp_latent = []
                        for k in range(args.neighbors):
                            temp_latent.append(all_add_latents[indices[g,k]][v]) #k,latent_size
                        temp_bs_latent.append(temp_latent)  #bs:k:latent_size
                    add_latent.append(temp_bs_latent)#num_layer:bs:k:lz
                add_latent = torch.from_numpy(np.array(add_latent)).to(args.device)
                similarity = torch.from_numpy(np.array(similarity)).to(args.device)
                similarity = torch.softmax(similarity,dim=-1)
            del inputs['texts']
            inputs['add_latent']=add_latent
            inputs['similarity'] = similarity
            model_output = model(**inputs)
            if args.use_bow:
                ce_loss, kl_loss, bow_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss + args.bow_weight * bow_loss
            else:
                ce_loss, kl_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss
            loss = loss.mean()
            valid_loss += loss.item()
            valid_ce_loss += ce_loss.mean().item()
            valid_kl_loss += kl_loss.mean().item()
            if args.use_bow:
                valid_bow_loss += bow_loss.mean().item()
        
        valid_loss = valid_loss / len(valid_iter)
        valid_ce_loss = valid_ce_loss / len(valid_iter)
        valid_kl_loss = valid_kl_loss / len(valid_iter)
        valid_bow_loss = valid_bow_loss / len(valid_iter)
        LOGGER.info('valid result: epoch {}, loss {}, ce_loss {}, kl {}, bow {}'.format(epoch, valid_loss, valid_ce_loss, valid_kl_loss, valid_bow_loss))
        
        if args.eval_metrics:
            ppl, elbo, nll, kl = calc_iwnll(model, valid_iter,index_list,all_add_latents,epoch,df_valid,ns=args.sample_times)
            au = calc_au(model, valid_iter)
            LOGGER.info('valid result: epoch {}, ppl {}, elbo {}, nll {}, kl {}'.format(epoch, ppl, elbo, nll, kl))
            LOGGER.info('valid result: epoch {}, au {}'.format(epoch, au))

def save(model, args, epoch):
    save_path = os.path.join(args.output_dir, args.model_name, 'model_epoch_{}.pt'.format(epoch))
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    try:
        model_to_save = model.module
    except:
        model_to_save = model
    torch.save(model_to_save.state_dict(), save_path)
    
def test(model, test_iter,args,df,tokenizer,LOGGER,beta=1):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_kl_loss = 0
        test_ce_loss = 0
        test_bow_loss = 0
        df = text_to_embeddings(df,model.encoder,tokenizer,args)
        index_list = []
        all_add_latents = df['latents'].values  
            
        for i in range(12):
            tmp_index = index_embeddings(df['latents'].values[:][i])
            index_list.append(tmp_index)
        for inputs in tqdm(test_iter, desc='test'):
            with torch.no_grad():
                query_output = model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            add_latent=[]
            similarity=[]
            tmp_latent = np.stack([i.cpu().numpy() for i in query_output.latent],axis=0)
            for v in range(12):
                query_vector=tmp_latent[v]
                distances, indices = index_list[v].search(query_vector, k = args.neighbors) #bs:k bs:k
                similarity.append(distances)#nl:bs:k
                temp_bs_latent=[]
                for g in range(len(indices)):
                    temp_latent = []
                    for k in range(args.neighbors):
                        temp_latent.append(all_add_latents[indices[g,k]][v]) #k,latent_size
                    temp_bs_latent.append(temp_latent)  #bs:k:latent_size
                add_latent.append(temp_bs_latent)
            add_latent = torch.from_numpy(np.array(add_latent)).to(args.device)
            similarity = torch.from_numpy(np.array(similarity)).to(args.device)
            similarity = torch.softmax(similarity,dim=-1)
            del inputs['texts']
            inputs['add_latent']=add_latent
            inputs['similarity'] = similarity
            model_output = model(**inputs)
            if args.use_bow:
                ce_loss, kl_loss, bow_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss + args.bow_weight * bow_loss
            else:
                ce_loss, kl_loss, _, _ = model_output
                loss = ce_loss + beta * kl_loss
            loss = loss.mean()
            test_loss += loss.item()
            test_ce_loss += ce_loss.mean().item()
            test_kl_loss += kl_loss.mean().item()
            if args.use_bow:
                test_bow_loss += bow_loss.mean().item()
        
        test_loss = test_loss / len(test_iter)
        test_ce_loss = test_ce_loss / len(test_iter)
        test_kl_loss = test_kl_loss / len(test_iter)
        test_bow_loss = test_bow_loss / len(test_iter)
        LOGGER.info('test result: loss {}, ce_loss {}, kl {}, bow {}'.format( test_loss, test_ce_loss, test_kl_loss, test_bow_loss))
        
        if args.eval_metrics:
            epoch=4
            df_valid=None
            ppl, elbo, nll, kl = calc_iwnll(model, test_iter,index_list,all_add_latents,epoch,df_valid,ns=args.sample_times)
            au = calc_au(model, test_iter)
            LOGGER.info('test result: ppl {}, elbo {}, nll {}, kl {}'.format(ppl, elbo, nll, kl))
            LOGGER.info('test result: au {}'.format(au))

def generate(model, test_iter, tokenizer, args,df): 
    if args.dataset_type == 'wp':
        has_condition = "conditional"
    else:
        has_condition = "unconditional"
    if args.top_k > 0:
        generate_param = "topk_{}".format(args.top_k)
    elif args.greedy_decoding:
        generate_param = "greedy_decoding"
    else:
        generate_param = "beamsearch_{}".format(args.num_beams)
    
    logging.info('{} generate with {}'.format(has_condition, generate_param))
    def filter_sen(sen):
        sen = sen.replace('<sep>', '')
        sen = sen.replace('<s>', '')
        sen = sen.replace('</s>', '')
        sen = sen.replace('<pad>', '')
        sen = sen.replace('<|endoftext|>', '')
        sen = sen.replace('<eos>', '')
        sen = ' '.join(sen.split())
        return sen
    model.eval()
    model.decoder.config.is_encoder_decoder = False

    output_list = []
    target_list = []
    source_list = []
    
    df = text_to_embeddings(df,model.encoder,tokenizer,args)
    index_list = []
    all_add_latents = df['latents'].values  #每个元素为num_layer:latent_size
            
    for i in range(12):
        tmp_index = index_embeddings(df['latents'].values[:][i])
        index_list.append(tmp_index)
    
    with torch.no_grad():
        for inputs in tqdm(test_iter):
            target = inputs['input_ids']
            if args.dataset_type == 'wp':
                source = inputs['condition']
            
            batch_size = target.size(0)
            device = target.device
            input_ids = target[:, 0].unsqueeze(1)
            model_kwargs = {}
            if args.dataset_type == 'wp':
                prior_latent = model.get_prior(batch_size, device, condition=inputs['condition'], condition_mask=inputs['condition_mask'])
                model_kwargs['attention_mask'] = inputs['condition_mask']
                input_ids = inputs['condition']
            else:
                prior_latent = model.get_prior(batch_size, device)
            
            tmp_latent = np.stack([i.cpu().numpy() for i in prior_latent],axis=0)
            
            add_latent=[]
            similarity=[]
                
            for v in range(12):
                query_vector=tmp_latent[v]
                distances, indices = index_list[v].search(query_vector, k = args.neighbors) #bs:k bs:k
                similarity.append(distances)#nl:bs:k
                temp_bs_latent=[]
                for g in range(len(indices)):
                    temp_latent = []
                    for k in range(args.neighbors):
                        temp_latent.append(all_add_latents[indices[g,k]][v]) #k,latent_size
                    temp_bs_latent.append(temp_latent)  #bs:k:latent_size
                add_latent.append(temp_bs_latent)#num_layer:bs:k:lz
            add_latent = torch.from_numpy(np.array(add_latent)).to(args.device)
            similarity = torch.from_numpy(np.array(similarity)).to(args.device)
            similarity = torch.softmax(similarity,dim=-1)
            # print(similarity.shape)
            # print(add_latent.shape)
            # del inputs['texts']
            # inputs['add_latent']=add_latent
            # inputs['similarity'] = similarity
            
            # print(add_latent)
            # print(similarity)
            gen_model = model.decoder
            if args.top_k > 0:
                ans = gen_model.generate(
                    input_ids, 
                    latent=prior_latent,
                    add_latent = add_latent,
                    similarity = similarity,
                    bos_token_id=tokenizer.bos_id, 
                    eos_token_id=tokenizer.eos_id, 
                    pad_token_id=tokenizer.pad_id, 
                    do_sample=True,
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    min_length=input_ids.size(-1) + 3, 
                    max_length=min(args.max_length, 1024),
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            elif args.greedy_decoding:
                ans = gen_model.generate(
                    input_ids, 
                    latent=prior_latent, 
                    add_latent = add_latent,
                    similarity = similarity,
                    bos_token_id=tokenizer.bos_id, 
                    eos_token_id=tokenizer.eos_id, 
                    pad_token_id=tokenizer.pad_id, 
                    min_length=input_ids.size(-1) + 3, 
                    max_length=min(args.max_length, 1024),
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            elif args.beam_search:
                ans = gen_model.generate(
                    input_ids, 
                    latent=prior_latent, 
                    add_latent = add_latent,
                    similarity = similarity,
                    bos_token_id=tokenizer.bos_id, 
                    eos_token_id=tokenizer.eos_id, 
                    pad_token_id=tokenizer.pad_id, 
                    num_beams=args.num_beams, 
                    min_length=input_ids.size(-1) + 3, 
                    max_length=min(args.max_length, 1024), 
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            else:
                if prior_latent is not None:
                    if isinstance(prior_latent, tuple):
                        latent = [item.repeat_interleave(args.num_beams, dim=0) for item in prior_latent]
                    else:
                        latent = prior_latent.repeat_interleave(args.num_beams, dim=0)
                else:
                    latent = None
                ans = gen_model.generate(
                    input_ids, 
                    latent=latent, 
                    bos_token_id=tokenizer.bos_id, 
                    eos_token_id=tokenizer.eos_id, 
                    pad_token_id=tokenizer.pad_id, 
                    num_beams=args.num_beams, 
                    min_length=input_ids.size(-1) + 3, 
                    max_length=min(args.max_length, 1024), 
                    repetition_penalty=args.repetition_penalty, 
                    **model_kwargs,
                )
            ans = ans.cpu().numpy()

            if args.dataset_type == 'wp':
                target = target.cpu().numpy()
                source = source.cpu().numpy()
            for i in range(len(ans)):
                text_ans = tokenizer.decode(ans[i], clean_up_tokenization_spaces=False)
                text_ans = filter_sen(text_ans)
                if len(text_ans) > 0:
                    output_list.append(text_ans)
                    if args.dataset_type in 'wp':
                        target_text = tokenizer.decode(target[i], clean_up_tokenization_spaces=False)
                        target_text = filter_sen(target_text)
                        target_list.append(target_text)
                        source_text = tokenizer.decode(source[i], clean_up_tokenization_spaces=False)
                        source_text = filter_sen(source_text)
                        source_list.append(source_text)

    save_dir = os.path.join(args.generation_output_dir, args.model_name)
    file_name = '{}_output_{}_epoch_{}_outputs.txt'.format(has_condition, generate_param, args.load_epoch)
    logging.info('generation output save at {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_name), 'w') as f:
        f.write('\n'.join(output_list))
    if args.dataset_type == 'wp':
        file_name = '{}_output_{}_epoch_{}_targets.txt'.format(has_condition, generate_param, args.load_epoch)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write('\n'.join(target_list))
        file_name = '{}_output_{}_epoch_{}_sources.txt'.format(has_condition, generate_param, args.load_epoch)
        with open(os.path.join(save_dir, file_name), 'w') as f:
            f.write('\n'.join(source_list))
