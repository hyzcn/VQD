EXPL="question and box model"
MODEL="boxonly"
NEPOCHS=15
NWORKERS=4
BS=128
echo 'Running all the experiments with same params .....'

python main.py --dataset refcoco  --evalsplit val  --save "val" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS


python main.py --dataset refcoco  --evalsplit testA --save "testA" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS


python main.py --dataset refcoco --evalsplit testB --save "testB"  --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS


python main.py --dataset refcoco+  --evalsplit val --save "val"  --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS

python main.py --dataset refcoco+  --evalsplit testA  --save "testA" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS

python main.py --dataset refcoco+  --evalsplit testB  --save "testB" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS

python main.py --dataset refcocog --evalsplit val  --save "val" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS

python main.py --dataset refcocog  --evalsplit test  --save "test" --batch_size $BS --model "$MODEL" --expl "$EXPL"  --workers $NWORKERS --epochs $NEPOCHS
