from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
import sys
import time

if __name__ == '__main__':
    mind_type = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    body_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    model = sys.argv[5] if len(sys.argv) > 5 else 'nrms'

    directory = '../data/{}/'.format(mind_type)
    yaml_file = directory + 'MIND{}_utils/{}.yaml'.format(mind_type, model)
    wordEmb_file = directory + 'MIND{}_utils/embedding.npy'.format(mind_type)
    wordDict_file = directory + 'MIND{}_utils/word_dict.pkl'.format(mind_type)
    userDict_file = directory + 'MIND{}_utils/uid2index.pkl'.format(mind_type)
    vertDict_file = directory + 'MIND{}_utils/vert_dict.pkl'.format(mind_type)
    subvertDict_file = directory + 'MIND{}_utils/subvert_dict.pkl'.format(mind_type)

    train_news_file = directory + 'MIND{}_train/news.tsv'.format(mind_type)
    train_behaviors_file = directory + 'MIND{}_train/behaviors.tsv'.format(mind_type)
    valid_news_file = directory + 'MIND{}_dev/news.tsv'.format(mind_type)
    valid_behaviors_file = directory + 'MIND{}_dev/behaviors.tsv'.format(mind_type)

    hparams = prepare_hparams(
        yaml_file=yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        body_size=body_size,
        epochs=epochs
    )
    print(hparams)

    model = NRMSModel(hparams, MINDAllIterator, seed=seed)

    pre_train_eval_res = model.run_eval(valid_news_file, valid_behaviors_file)
    print(f'\n\nPre-train evaluation results:\n{pre_train_eval_res}\n\n')

    model.fit(
        train_news_file=train_news_file,
        train_behaviors_file=train_behaviors_file,
        valid_news_file=valid_news_file,
        valid_behaviors_file=valid_behaviors_file
    )
    model.model.save_weights(directory + f'/weights_{int(time.time())}.h5')

    post_train_eval_res = model.run_eval(valid_news_file, valid_behaviors_file)
    print(f'\n\nPost-train evaluation results:\n{post_train_eval_res}\n\n')



