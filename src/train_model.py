from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
# from recommenders.models.newsrec.models.naml import NAMLModel
from naml.naml import NAMLModel
import sys
import time

if __name__ == '__main__':
    mind_type = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    body_size = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    model_name = sys.argv[6] if len(sys.argv) > 6 else 'nrms'

    directory = f'../data/{mind_type}/MIND{mind_type}_'
    yaml_file = directory + f'utils/{model_name}.yaml'
    wordEmb_file = directory + 'utils/embedding.npy'
    wordDict_file = directory + 'utils/word_dict.pkl'
    userDict_file = directory + 'utils/uid2index.pkl'
    vertDict_file = directory + 'utils/vert_dict.pkl'
    subvertDict_file = directory + 'utils/subvert_dict.pkl'

    train_news_file = directory + 'train/news.tsv'
    train_behaviors_file = directory + 'train/behaviors.tsv'
    # valid_news_file = directory + 'dev/news.tsv'
    # valid_behaviors_file = directory + 'dev/behaviors.tsv'
    valid_news_file = f'../data/demo/MINDdemo_dev/news.tsv'
    valid_behaviors_file = f'../data/demo/MINDdemo_dev/behaviors.tsv'

    hparams = prepare_hparams(
        yaml_file=yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        batch_size=batch_size,
        body_size=body_size,
        epochs=epochs,
        save_model=True,
        save_epoch=1,
        show_step=1000
    )
    print(hparams)

    model = None
    if model_name == 'nrms':
        model = NRMSModel(hparams, MINDAllIterator, seed=seed)
    elif model_name == 'naml':
        model = NAMLModel(hparams, MINDAllIterator, seed=seed)
    else:
        raise ValueError(f'Model {model_name} not supported')

    # pre_train_eval_res = model.run_eval(valid_news_file, valid_behaviors_file)
    # print(f'\n\nPre-train evaluation results:\n{pre_train_eval_res}\n\n')

    model.fit(
        train_news_file=train_news_file,
        train_behaviors_file=train_behaviors_file,
        valid_news_file=valid_news_file,
        valid_behaviors_file=valid_behaviors_file
    )
    model.model.save_weights(f'../data/{mind_type}/weights_{model_name}_{int(time.time())}.h5')

    # post_train_eval_res = model.run_eval(valid_news_file, valid_behaviors_file)
    # print(f'\n\nPost-train evaluation results:\n{post_train_eval_res}\n\n')

# step 2000 , total_loss: 1.4525, data_loss: 1.1757
# step 3000 , total_loss: 1.4291, data_loss: 1.4575
# step 4000 , total_loss: 1.4128, data_loss: 1.3653
# step 5000 , total_loss: 1.3987, data_loss: 1.2809
# step 6000 , total_loss: 1.3875, data_loss: 1.3230
# step 7000 , total_loss: 1.3793, data_loss: 1.2980
# step 9000 , total_loss: 1.3663, data_loss: 1.2288