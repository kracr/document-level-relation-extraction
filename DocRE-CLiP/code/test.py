import sklearn.metrics
import torch

from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.DRN import DRN_GloVe, DRN_BERT
from utils import get_cuda, logging, print_params,logging, print_params, Metrics
from torch import nn
from collections import  Counter,defaultdict
import pickle
from RGCN import *
from RGCN.models import RGCN
from RGCN.utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
import ast
import json

dirlinkpredict='/home/DocRE-CLiP/data/DOCRED/DOCREDlink/linkpredictiondata/'
def linkprediction():

    best_mrr = 0

    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data(dirlinkpredict)
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    print("all triples length:",len(all_triplets))
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)

    model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint = torch.load('/home/DocRE-CLiP/code/checkpoint/best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])


    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    return entity_embedding,model.relation_embedding


def parsed(file_path):
    #file_path = "final.json"
    with open(file_path, "r") as file:
       json_data1 = file.read()

    parsed_data = ast.literal_eval(json_data1)
    return parsed_data


def readdict():
    with open(dirlinkpredict+'entities.dict') as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(dirlinkpredict+'relations.dict') as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    return entity2id,relation2id 



def test(model, dataloader, modelname, id2rel, output_file=False, test_prefix='dev',lr_rate=0,global_step=0,config=None):
    # ours: inter-sentence F1 in LSR

    import gc

    gc.collect()

    torch.cuda.empty_cache()
    relation_num = config.relation_nums
    #print("relation num",relation_num)
    input_theta = config.input_theta

    BCELogit = nn.BCEWithLogitsLoss(reduction='none')

    final_result=[]
    test_result = []
    test_metric = Metrics("Re Test",logging,use_wandb=config.use_wandb)
    test_metric.reset()

    theta_list = input_theta*np.ones((1,relation_num),np.float32)
    scorelist=[]
    entity_embedding,relation_embedding=linkprediction()
    #scoredata=parsed("key.json")
    ent_dict,rel_dict=readdict()


    for cur_i, d in enumerate(dataloader):
        # print('step: {}/{}'.format(cur_i, total_steps))
        #print(d)

        with torch.no_grad():
            relation_multi_label = d['relation_multi_label']
            ign_relation_multi_label = d['ign_relation_multi_label']
            relation_mask = d['relation_mask']
            labels = d['labels']
            L_vertex = d['L_vertex']
            titles = d['titles']
            indexes = d['indexes']
            overlaps = d['overlaps']
            relation_path = d["relation_path"]
            #print("RELATION_PATH:",relation_path)
            #print("labels:",labels)

            output = model( words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                context_ems_info=d['context_ems_info'],
                                h_t_pairs=d['h_t_pairs'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                sentence_id=d["context_sent"],
                                mention_id=d["context_mention"],
                                relation_mask=relation_mask,
                                ht_pair_distance=d['ht_pair_distance'],
                                ht_sent_distance=d["ht_sent_distance"],
                                graph_adj=d['graph_adj'],
                                graph_info=d["graph_info"],
                                graph_node_num=d["graph_node_num"],
                                relation_path=d["relation_path"],
                                L_vertex=d["L_vertex"],
                                titles=d["titles"],
                                entity_embedding=entity_embedding,
                                relation_embedding=relation_embedding,
                                ent_dict=ent_dict,
                                rel_dict=rel_dict,
                                modeltype="test"
                                )
            predictions = output["predictions"]
            loss = torch.sum(BCELogit(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    relation_num * torch.sum(relation_mask))
            
            #relationscore=output["scores"]
            ## Relation
            #relationscore=output["scores"]
            ## Relation
            scores = predictions

            #print(scores)
            #scorelist.append(scores)
            total_mask = relation_mask>0
           
            test_metric.roc_record(loss,scores[...,1:],relation_multi_label[...,1:],total_mask,ign=ign_relation_multi_label[...,1:])
            final_result=[]

            if output_file:
                scores = predictions.data.cpu().numpy()
                #newscore = relationscore.data.cpu().numpy()

                

                for i in range(len(titles)):
                    j = 0
                    for h_idx in range(L_vertex[i]):
                        for t_idx in range(L_vertex[i]):
                            if h_idx == t_idx:
                                continue
                            for label_idx in range(1,relation_num):
                                score_value = scores[i, j, label_idx]
                                score_tensor = torch.tensor(score_value)
                                probability = torch.sigmoid(score_tensor)
                                probability_np = probability.item()
                                rounded = np.round(probability_np, decimals=6)


                                #print(rounded)
                                final_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":id2rel[label_idx],"score":rounded})

                                #rounded_scores = np.round(probability_np, decimals=2)
                                #print("title",titles[i],"h_idx",h_idx,"t_idx",t_idx,"r",id2rel[label_idx],"score",rounded_scores)

                                #final_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":id2rel[label_idx],"score":rounded[i,j,label_idx]})
                                #print(theta_list[0,label_idx-1])
                                #print(h_idx,t_idx,label_idx,scores[i,j,label_idx])
                                #if scores[i,j,label_idx]>theta_list[0,label_idx-1]:
                                    #print(h_idx,t_idx,"score:", scores[i,j,label_idx])
                                    #test_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":id2rel[label_idx]})
                                    #score_value = scores[i, j, label_idx]
                                    #score_tensor = torch.tensor(score_value)
                                    #probability = torch.sigmoid(score_tensor)
                                    #probability_np = probability.item()
                                    #rounded = np.round(probability_np, decimals=4)
                                    #print("title",titles[i],"h_idx",h_idx,"t_idx",t_idx,"r",id2rel[label_idx],"score",rounded)

                                #     final_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":id2rel[label_idx],"score":rounded})

                            j+=1
                            


    loss,test_acc,test_recall,test_ign_f1,test_f1,theta = test_metric.cal_roc_metric(global_step,lr_rate,log=True)

    #if output_file:
    #    json.dump(test_result, open(test_prefix + "_index.json", "w"))
    
    #    print("result file finish")
    #json.dump(final_result, open(test_prefix + "_final_full_index.json", "w"))
    #np.save("scores.npy",scorelist)

    return loss,test_ign_f1, test_f1,theta, test_acc, test_recall


if __name__ == '__main__':
    opt = get_opt()
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    print(json.dumps(opt.__dict__, indent=4))
    rel2id = json.load(open(os.path.join(opt.data_dir, 'rel2id.json'), "r"))
    id2rel = {v: k for k, v in rel2id.items()}
    word2id = json.load(open(os.path.join(opt.data_dir, 'word2id.json'), "r"))
    ner2id = json.load(open(os.path.join(opt.data_dir, 'ner2id.json'), "r"))
    opt.data_word_vec = np.load(os.path.join(opt.data_dir, 'vec.npy'))


    #export CUDA_VISIBLE_DEVICES=$1

    input_theta=2--1
    batch_size=5
    test_batch_size=5
    #dataset=dev

    opt.model_name=DRN_BERT


    dataset="docred" 
    opt.use_model= "bert" 
    opt.pretrain_model= "checkpoint/DRN_BERT_base_200.pt" 
    gcn_dim =808 
    gcn_layers= 2 
    bert_hid_size= 768 
    bert_path ="bert-base-uncased" 
    use_entity_type=True
    use_entity_id =True
    use_graph =True
    use_dis_embed =True
    use_context =True
    graph_type= "gcn" 
    activation ="relu" 
    test_type="dev"


    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',opt=opt)
        if opt.test_type == "dev":
            test_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        else:
            test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, opt, batch_size=opt.test_batch_size, dataset_type='test')
        print(opt.model_name)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #if "DRN" in opt.model_name:
        model = DRN_BERT(opt)
        #elif "MPR" in opt.model_name:
        #model = MPR_BERT(opt)
        #else:
        #    raise("Error")
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',opt=opt)
        if opt.test_type == "dev":
            test_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        else:
            test_set = DGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, opt,batch_size=opt.test_batch_size, dataset_type='test')

        print(model_name)
        print("############################")
        print(opt)
        #if "DRN" in opt.model_name:
        model = DRN_GloVe(opt)
        #elif "MPR" in opt.model_name:
        #    model = MPR_GloVe(opt)
        #else:
        #    raise("Error")
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    import gc

    del train_set
    gc.collect()

    print(model.parameters)
    #print_params(opt.model)
    opt.model="bert"
    start_epoch = 1
    pretrain_model = "checkpoint/DRN_BERT_base_200.pt"
    lr = opt.lr
    model_name = "DRN_BERT"

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model = get_cuda(model)
    model.eval()

    
    
    loss,ign_f1, f1, theta, pr_x, pr_y = test(model, test_loader, model_name, id2rel=id2rel,
                            output_file=True, test_prefix='test',config=opt)
    print('finished')
    
