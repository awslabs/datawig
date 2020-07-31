class SimpleImputer {
  // string, string
  constructor(input_col, output_col) {
    this.input_col = input_col
    this.output_col = output_col

    this.vectorizer = new BinaryVectorizer(this.input_col)
    this.label_encoder = new LabelEncoder(this.output_col)

    // TODO: partial_fit
    // this.X = null
    // this.y = null

    this.A = null
    this.model = null

    this.partial_fit_iteration = 0
  }

  // explain_instance(instance, k) { }

  compute_explain_patterns(dataset) {
    const X = this.vectorizer.transform(dataset)
    const d = X.shape[1]

    // only compute first time
    if (this.A == null) {
      const scaler = new StandardScaler()
      const X_normalized = scaler.fit_transform(X)
      const yhat_normalized = scaler.fit_transform(this.model.predict(X))
      // pattern: dxk
      this.A = X_normalized.transpose().dot(yhat_normalized)
    }
  }

  // class_index: int, k: int
  explain(class_index, k) {
    const d = this.vectorizer.tokens.size//this.X.shape[1]

    const A_ = this.A.dataSync()
    const patternForIndex = new Array(this.A.shape[0])
    for(let i = 0; i < this.A.shape[0]; i++) {
      patternForIndex[i] = A_[(i*this.A.shape[1]) + class_index]
    }
    const patternForIndexT = tf.tensor2d(patternForIndex, [1, d])

    patternForIndexT.print()

    // array of length d with tokens
    const label_tokens = this.vectorizer.inverse_transform(patternForIndexT)[0]

    // sorted indices descending by pattern magnitude
    const pattern_args_sorted = Math_.argsort(patternForIndex, false)

    console.log('Explaining class index', class_index, '(', this.label_encoder.index_labels[class_index], ') with', k, 'tokens')
    const top_k_tokens = new Array()
    // print top K tokens with importances
    for (let i = 0; i < k; i++) {
      const idx = pattern_args_sorted[i]
      const importance = patternForIndex[idx]
      if (importance > 0) {
        console.log(label_tokens[idx], importance)
        top_k_tokens.push({token: label_tokens[idx], importance: importance})
      }
    }
    return top_k_tokens
  }

  // array[obj]
  predict_proba(dataset) {
    const X = this.vectorizer.transform(dataset)
    return this.model.predict(X)
  }

  partial_fit(dataset, validationSplit = 0.25, nEpochs = 10) {
    // const trainData = new Array()
    // const valData = new Array()
    //
    // for(const element of dataset) {
    //   if (Math.random() <= validationSplit) {
    //     valData.push(element)
    //   } else {
    //     trainData.push(element)
    //   }
    // }
    //
    // if (valData.length == 0) {
    //   return
    // }

    const trainData = dataset;
    if (!this.vectorizer.is_fitted()) {
      this.vectorizer.fit(trainData)
    }
    if (!this.label_encoder.is_fitted()) {
      this.label_encoder.fit(trainData)
    }

    const X = this.vectorizer.transform(trainData)
    const y = this.label_encoder.transform(trainData)
    // const X_val = this.vectorizer.transform(valData)
    // const y_val = this.label_encoder.transform(valData)

    console.log(X.shape, y.shape)//, X_val.shape, y_val.shape)

    const d = X.shape[1]
    const k = y.shape[1]

    if (this.model == null) {
      // https://js.tensorflow.org/api/latest/#layers.dense
      this.model = tf.sequential()
      this.model.add(
        tf.layers.dense({name: 'inputs', inputDim: d, units: k, activation: 'softmax', useBias: true})
      )

      this.model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.sgd(0.25)
      })
    }

    // learning rate decay
    const initial_lr = 0.25
    const lr_decay = 0.05 // 0.01
    const lr = initial_lr * (1 / (1 + lr_decay * this.partial_fit_iteration))
    this.partial_fit_iteration++
    console.log('Updating learning rate from', this.model.optimizer_.learningRate, 'to', lr)
    this.model.optimizer_.learningRate = lr

    const historyPromise = this.model.fit(X, y, {
      epochs: nEpochs,
      batchSize: 100,
      verbose: 2
      // callbacks: {
      //   onEpochEnd: async (epoch, logs) => {
      //     const pred = this.model.predict(X_val)
      //
      //     // https://github.com/tensorflow/tfjs-examples/blob/651fd08a0a7088e6bbb5fb6b8f3816b1c23ec74f/website-phishing/utils.js#L175
      //     const pred_eq_max = pred.max(1).broadcastTo([pred.shape[1], pred.shape[0]]).transpose().equal(pred)
      //     const binarized = tf.where(pred_eq_max, tf.onesLike(pred), tf.zerosLike(pred))
      //     // pred.print()
      //     // binarized.print()
      //     const precision = tf.metrics.precision(y_val, binarized).dataSync()*100.0
      //     const recall = tf.metrics.recall(y_val, binarized).dataSync()*100.0
      //     const log = `Epoch ${epoch}: Precision ${precision.toFixed(2)}% and Recall ${recall.toFixed(2)}%`
      //     console.log(log)
      //   }
      // }
    })

    return historyPromise
  }

  // // TODO: partial fit transform for dataset
  // // array[obj], double, int
  // fit(dataset, validationSplit = 0.25, nEpochs = 50) {
  //   const trainData = new Array()
  //   const valData = new Array()
  //
  //   for(const element of dataset) {
  //     if (Math.random() <= validationSplit) {
  //       valData.push(element)
  //     } else {
  //       trainData.push(element)
  //     }
  //   }
  //
  //   console.log('Fitting on', trainData.length, 'rows')
  //   console.log('With', valData.length, 'rows of validation data')
  //
  //   this.X = this.vectorizer.fit_transform(trainData)
  //   this.y = this.label_encoder.fit_transform(trainData)
  //   const d = this.X.shape[1]
  //   const k = this.y.shape[1]
  //
  //   const X_val = this.vectorizer.transform(valData)
  //   const y_val = this.label_encoder.transform(valData)
  //
  //   // https://js.tensorflow.org/api/latest/#layers.dense
  //   this.model = tf.sequential()
  //   this.model.add(
  //     tf.layers.dense({name: 'inputs', inputDim: d, units: k, activation: 'softmax', useBias: true})
  //   )
  //
  //   this.model.compile({
  //     loss: 'categoricalCrossentropy',
  //     optimizer: tf.train.sgd(0.25)
  //     // metrics: [tf.metrics.precision]
  //     // metrics: ['accuracy', 'precision']
  //   })
  //
  //   const history = this.model.fit(this.X, this.y, {
  //     epochs: nEpochs,
  //     batchSize: 100,
  //     shuffle: true,
  //     // validationSplit: 0.1,
  //     validationData: [X_val, y_val],
  //     verbose: 2,
  //     // callbacks: tf.callbacks.earlyStopping({monitor: 'val_loss'})
  //     // callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
  //     // callbacks: {onEpochEnd: onEpochEnd}
  //     callbacks: {
  //       onEpochEnd: async (epoch, logs) => {
  //         const pred = this.model.predict(X_val)
  //
  //         // https://github.com/tensorflow/tfjs-examples/blob/651fd08a0a7088e6bbb5fb6b8f3816b1c23ec74f/website-phishing/utils.js#L175
  //         const pred_eq_max = pred.max(1).broadcastTo([pred.shape[1], pred.shape[0]]).transpose().equal(pred)
  //         const binarized = tf.where(pred_eq_max, tf.onesLike(pred), tf.zerosLike(pred))
  //         // pred.print()
  //         // binarized.print()
  //         const precision = tf.metrics.precision(y_val, binarized).dataSync()*100.0
  //         const recall = tf.metrics.recall(y_val, binarized).dataSync()*100.0
  //         const log = `Epoch ${epoch}: Precision ${precision.toFixed(2)}% and Recall ${recall.toFixed(2)}%`
  //         console.log(log)
  //
  //         $('#training_stats').empty()
  //         $('#training_stats').append(log)
  //
  //         // const progress = 100.0*(epoch+1)/nEpochs
  //         // $('#progressbar').progressbar(
  //         //   {value: progress}
  //         // )
  //       }
  //     }
  //   })
  //
  // }

  /**
   * Selects at most k samples that have the lagest uncertainty (as measured by uncertainty sampling).
   *
   * array[obj], int
   */
  getTopKMostUncertainSamples(dataset, k) {
    const maxScores = this.predict_proba(dataset).max(1).arraySync()
    const maxScoreIndices = Math_.argsort(maxScores, true)
    const mostUncertainSamples = new Array(k)
    let i = 0
    for (let idx of maxScoreIndices.slice(0, k)) {
      mostUncertainSamples[i] = dataset[idx]
      mostUncertainSamples[i]['uncertainty'] = 1 - maxScores[idx]
      // mostUncertainSamples[i]['dataset_index'] = idx
      i++
    }
    return mostUncertainSamples
  }
}
