data_dir=simple-examples

if [ ! -d $data_dir ]; then
  wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  tar xfz simple-examples.tgz
  rm simple-examples.tgz

  mv simple-examples/data/ptb.train.txt ./
  rm -rf simple-examples
fi
