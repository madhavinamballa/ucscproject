{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, LSTM, Dropout, Activation\n",
    "from keras.optimizers import RMSprop, Adam\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "from keras.utils import np_utils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "SEQ_LENGTH = 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def buildmodel(VOCABULARY):\n",
    "    model = Sequential()\n",
    "    model.add(LSTM(256, input_shape = (SEQ_LENGTH, 1), return_sequences = True))\n",
    "    model.add(Dropout(0.2))\n",
    "    model.add(LSTM(256))\n",
    "    model.add(Dropout(0.2))\n",
    "    model.add(Dense(VOCABULARY, activation = 'softmax'))\n",
    "    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "file = open('wonderland.txt', encoding = 'utf8')\n",
    "raw_text = file.read()    #you need to read further characters as well\n",
    "raw_text = raw_text.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['\\n', ' ', '!', '(', ')', '*', ',', '-', '.', ':', ';', '?', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ù', '—', '‘', '’', '“', '”']\n",
      "=====================\n",
      "['\\n', ' ', '!', '(', ')', ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ù', '—', '‘', '’', '“', '”']\n",
      "43\n",
      "22999\n"
     ]
    }
   ],
   "source": [
    "chars = sorted(list(set(raw_text)))\n",
    "print(chars)\n",
    "\n",
    "bad_chars = ['#', '*', '@', '_', '\\ufeff']\n",
    "for i in range(len(bad_chars)):\n",
    "    raw_text = raw_text.replace(bad_chars[i],\"\")\n",
    "\n",
    "chars = sorted(list(set(raw_text)))\n",
    "print(\"=====================\")\n",
    "print(chars)\n",
    "print(len(chars))\n",
    "print(len(raw_text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Text length = 22999\n",
      "No. of characters = 43\n"
     ]
    }
   ],
   "source": [
    "text_length = len(raw_text)\n",
    "char_length = len(chars)\n",
    "VOCABULARY = char_length\n",
    "print(\"Text length = \" + str(text_length))\n",
    "print(\"No. of characters = \" + str(char_length))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "char_to_int = dict((c, i) for i, c in enumerate(chars))\n",
    "int_to_char = dict((i, c) for i, c in enumerate(chars))\n",
    "input_strings = []\n",
    "output_strings = []\n",
    "\n",
    "for i in range(len(raw_text) - SEQ_LENGTH):\n",
    "    X_text = raw_text[i: i + SEQ_LENGTH]\n",
    "    X = [char_to_int[char] for char in X_text]\n",
    "    input_strings.append(X)    \n",
    "    Y = raw_text[i + SEQ_LENGTH]\n",
    "    output_strings.append(char_to_int[Y])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "22899\n",
      "(22899, 100, 1)\n",
      "(22899, 43)\n"
     ]
    }
   ],
   "source": [
    "length = len(input_strings)\n",
    "input_strings = np.array(input_strings)\n",
    "print(length)\n",
    "input_strings = np.reshape(input_strings, (input_strings.shape[0], input_strings.shape[1], 1))\n",
    "print(input_strings.shape)\n",
    "input_strings = np.reshape(input_strings, (input_strings.shape[0], input_strings.shape[1], 1))\n",
    "input_strings = input_strings/float(VOCABULARY)\n",
    "\n",
    "output_strings = np.array(output_strings)\n",
    "output_strings = np_utils.to_categorical(output_strings)\n",
    "# print(input_strings.shape)\n",
    "print(output_strings.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 3.0552\n",
      "Epoch 00001: loss improved from inf to 3.05522, saving model to saved_models/weights-improvement-01-3.0552.hdf5\n",
      "179/179 [==============================] - 103s 576ms/step - loss: 3.0552\n",
      "Epoch 2/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 3.0110\n",
      "Epoch 00002: loss improved from 3.05522 to 3.01102, saving model to saved_models/weights-improvement-02-3.0110.hdf5\n",
      "179/179 [==============================] - 103s 577ms/step - loss: 3.0110\n",
      "Epoch 3/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.9652\n",
      "Epoch 00003: loss improved from 3.01102 to 2.96522, saving model to saved_models/weights-improvement-03-2.9652.hdf5\n",
      "179/179 [==============================] - 103s 575ms/step - loss: 2.9652\n",
      "Epoch 4/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.8609\n",
      "Epoch 00004: loss improved from 2.96522 to 2.86094, saving model to saved_models/weights-improvement-04-2.8609.hdf5\n",
      "179/179 [==============================] - 103s 578ms/step - loss: 2.8609\n",
      "Epoch 5/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.7553\n",
      "Epoch 00005: loss improved from 2.86094 to 2.75530, saving model to saved_models/weights-improvement-05-2.7553.hdf5\n",
      "179/179 [==============================] - 103s 578ms/step - loss: 2.7553\n",
      "Epoch 6/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.6925\n",
      "Epoch 00006: loss improved from 2.75530 to 2.69255, saving model to saved_models/weights-improvement-06-2.6925.hdf5\n",
      "179/179 [==============================] - 103s 578ms/step - loss: 2.6925\n",
      "Epoch 7/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.6392\n",
      "Epoch 00007: loss improved from 2.69255 to 2.63917, saving model to saved_models/weights-improvement-07-2.6392.hdf5\n",
      "179/179 [==============================] - 104s 578ms/step - loss: 2.6392\n",
      "Epoch 8/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.5857  \n",
      "Epoch 00008: loss improved from 2.63917 to 2.58571, saving model to saved_models/weights-improvement-08-2.5857.hdf5\n",
      "179/179 [==============================] - 16833s 94s/step - loss: 2.5857\n",
      "Epoch 9/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.5359 \n",
      "Epoch 00009: loss improved from 2.58571 to 2.53588, saving model to saved_models/weights-improvement-09-2.5359.hdf5\n",
      "179/179 [==============================] - 2582s 14s/step - loss: 2.5359\n",
      "Epoch 10/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.4812  \n",
      "Epoch 00010: loss improved from 2.53588 to 2.48121, saving model to saved_models/weights-improvement-10-2.4812.hdf5\n",
      "179/179 [==============================] - 28421s 159s/step - loss: 2.4812\n",
      "Epoch 11/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.4271\n",
      "Epoch 00011: loss improved from 2.48121 to 2.42708, saving model to saved_models/weights-improvement-11-2.4271.hdf5\n",
      "179/179 [==============================] - 308s 2s/step - loss: 2.4271\n",
      "Epoch 12/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.3752\n",
      "Epoch 00012: loss improved from 2.42708 to 2.37520, saving model to saved_models/weights-improvement-12-2.3752.hdf5\n",
      "179/179 [==============================] - 311s 2s/step - loss: 2.3752\n",
      "Epoch 13/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.3265\n",
      "Epoch 00013: loss improved from 2.37520 to 2.32654, saving model to saved_models/weights-improvement-13-2.3265.hdf5\n",
      "179/179 [==============================] - 318s 2s/step - loss: 2.3265\n",
      "Epoch 14/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.2735\n",
      "Epoch 00014: loss improved from 2.32654 to 2.27353, saving model to saved_models/weights-improvement-14-2.2735.hdf5\n",
      "179/179 [==============================] - 184s 1s/step - loss: 2.2735\n",
      "Epoch 15/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.2200\n",
      "Epoch 00015: loss improved from 2.27353 to 2.22000, saving model to saved_models/weights-improvement-15-2.2200.hdf5\n",
      "179/179 [==============================] - 115s 642ms/step - loss: 2.2200\n",
      "Epoch 16/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.1694\n",
      "Epoch 00016: loss improved from 2.22000 to 2.16939, saving model to saved_models/weights-improvement-16-2.1694.hdf5\n",
      "179/179 [==============================] - 104s 578ms/step - loss: 2.1694\n",
      "Epoch 17/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.1174\n",
      "Epoch 00017: loss improved from 2.16939 to 2.11742, saving model to saved_models/weights-improvement-17-2.1174.hdf5\n",
      "179/179 [==============================] - 107s 596ms/step - loss: 2.1174\n",
      "Epoch 18/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.0677\n",
      "Epoch 00018: loss improved from 2.11742 to 2.06767, saving model to saved_models/weights-improvement-18-2.0677.hdf5\n",
      "179/179 [==============================] - 104s 583ms/step - loss: 2.0677\n",
      "Epoch 19/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 2.0183\n",
      "Epoch 00019: loss improved from 2.06767 to 2.01830, saving model to saved_models/weights-improvement-19-2.0183.hdf5\n",
      "179/179 [==============================] - 106s 592ms/step - loss: 2.0183\n",
      "Epoch 20/20\n",
      "179/179 [==============================] - ETA: 0s - loss: 1.9614\n",
      "Epoch 00020: loss improved from 2.01830 to 1.96143, saving model to saved_models/weights-improvement-20-1.9614.hdf5\n",
      "179/179 [==============================] - 102s 572ms/step - loss: 1.9614\n"
     ]
    }
   ],
   "source": [
    "model = buildmodel(VOCABULARY)\n",
    "filepath=\"saved_models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5\"\n",
    "checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')\n",
    "callbacks_list = [checkpoint]\n",
    "\n",
    "history = model.fit(input_strings, output_strings, epochs = 20, batch_size = 128, callbacks = callbacks_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = 'saved_models/weights-improvement-20-1.9614.hdf5'\n",
    "model = buildmodel(VOCABULARY)\n",
    "model.load_weights(filename)\n",
    "model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 30, 18, 15, 1, 29, 31, 24, 1, 14, 19, 14, 1, 24, 25, 30, 1, 29, 18, 19, 24, 15, 5, 1, 19, 30, 1, 33, 11, 29, 1, 30, 25, 25, 1, 33, 15, 30, 1, 30, 25, 1, 26, 22, 11, 35, 5, 1, 29, 25, 1, 33, 15, 1, 29, 11, 30, 1, 19, 24, 1, 18, 15, 28, 15, 1, 33, 15, 1, 30, 33, 25, 1, 11, 24, 14, 1, 33, 15, 1, 29, 11, 19, 14, 1, 18, 25, 33, 1, 33, 15, 1, 33, 19, 29, 18, 1, 33, 15, 1, 18, 11, 14, 1, 29, 25, 23, 15, 30, 18, 19, 24, 17, 1, 30, 25, 1, 14, 25, 7]\n"
     ]
    }
   ],
   "source": [
    "initial_text = ' the sun did not shine, it was too wet to play, so we sat in here we two and we said how we wish we had something to do.'\n",
    "initial_text = [char_to_int[c] for c in initial_text]\n",
    "print(initial_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "initial_text1 = 'madhavi namballa'\n",
    "initial_text1 = [char_to_int[c] for c in initial_text1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "GENERATED_LENGTH = 10\n",
    "test_text = initial_text1\n",
    "generated_text = []\n",
    "int_to_char = dict((i, c) for i, c in enumerate(chars))\n",
    "\n",
    "for i in range(GENERATED_LENGTH):\n",
    "    X = np.reshape(test_text, (1,16,1))\n",
    "    next_character = model.predict(X/float(VOCABULARY))\n",
    "    index = np.argmax(next_character)\n",
    "    generated_text.append(int_to_char[index])\n",
    "    test_text.append(index)\n",
    "    test_text = test_text[1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " and she w\n"
     ]
    }
   ],
   "source": [
    "print(''.join(generated_text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernel_info": {
   "name": "dev"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  },
  "nteract": {
   "version": "0.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
