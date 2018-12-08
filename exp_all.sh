set -e

get_all=$1
src=$2

d=$(dirname "$0")

if [[ $src = 'es' ]]; then
    id=kg4j3u604o
    #year=13
    tw="buen bueno"
elif [[ $src = 'de' ]]; then
    id=1dghwzosvk
    #year=17
    tw="gute Tag"
fi

if [[ $get_all -eq 1 ]]; then

    python $d/exp.py --exp_dir dumped/debug/$id/ --src_emb_path /data/rsg/nlp/j_luo/cc.$src.300.vec --tgt_emb_path /data/rsg/nlp/j_luo/cc.en.300.vec --src_lang $src --tgt_lang en --emb_dim 300 --test_words $tw --data_dir europarl/$src 
    #python $d/exp.py --exp_dir dumped/debug/otsvbxjpc4/ --src_emb_path /data/rsg/nlp/j_luo/cc.$src.300.vec --tgt_emb_path /data/rsg/nlp/j_luo/cc.en.300.vec --src_lang $src --tgt_lang en --emb_dim 300 --test_words buen bueno --data_dir wmt13-$src-100K --center

elif [[ $get_all -eq 0  ]]; then
    src_vocabs=$(ls ../EMNLP-NMT/data/europarl/$src/vocab.*.$src*)
    tgt_vocabs=$(ls ../EMNLP-NMT/data/europarl/$src/vocab.*.en*)
    python $d/exp.py --exp_dir dumped/debug/$id/ --src_emb_path /data/rsg/nlp/j_luo/cc.$src.300.vec --tgt_emb_path /data/rsg/nlp/j_luo/cc.en.300.vec --src_lang $src --tgt_lang en --emb_dim 300 --test_words $tw --data_dir europarl/$src --src_vocab_path $src_vocabs --tgt_vocab_path $tgt_vocabs
    #python $d/exp.py --exp_dir dumped/debug/otsvbxjpc4/ --src_emb_path /data/rsg/nlp/j_luo/cc.$src.300.vec --tgt_emb_path /data/rsg/nlp/j_luo/cc.en.300.vec --src_lang $src --tgt_lang en --emb_dim 300 --test_words buen bueno --data_dir wmt13-$src-100K --src_vocab_path $src_vocabs --tgt_vocab_path $tgt_vocabs --center

fi
