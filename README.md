# GD.BERT-pretrained-model-
# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 양주영
- 리뷰어 : 김민식


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 네. 루브릭의 평가문항들을 충족하였습니다. (GPU 부족 등 현실적 문제 감안 필요)
    '''
    n_seq = 10

    # make test inputs
    enc_tokens = np.random.randint(0, len(vocab), (10, n_seq))
    segments = np.random.randint(0, 2, (10, n_seq))
    labels_nsp = np.random.randint(0, 2, (10,))
    labels_mlm = np.random.randint(0, len(vocab), (10, n_seq))
    
    test_model = build_model_pre_train(config)
    test_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),         metrics=["acc"])
    
    # test model fit
    test_model.fit((enc_tokens, segments), (labels_nsp, labels_mlm), epochs=2, batch_size=5)

- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 네. doc string이 task별로 잘 작성돼 이해하기 잘 됐습니다.
    '''
        # 위키가 주제별로 잘 나눠지는지 여부 확인
        count = 5
        
        with open(corpus_file, 'r') as in_f:
            doc = []  # 단락 단위로 문서 저장
            for line in tqdm(in_f, total=total):
                line = line.strip()
                if line == "":  # line이 빈줄 일 경우 (새로운 단락)  
                    if 0 < len(doc):
                        if 0 < count:
                            count -= 1
                            print(len(doc), "lines :", doc[0])
                            print(doc[1])
                            print(doc[-1])
                            print()
                        else:
                            break
                        doc = []
                else:  # 빈 줄이 아니면 doc에 저장
                    pieces = vocab.encode_as_pieces(line)    
                    if 0 < len(pieces):
                        doc.append(pieces)
            if 0 < len(doc):  # 마지막에 처리되지 않은 doc가 있는 경우
                print(doc[0])
                print(doc[1])
                print(doc[-1])
                doc = []
    '''
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
  
- [X]  **4. 회고를 잘 작성했나요?**
    - 네 문제사항과 개선점에 대해서 상세히 기재되어 있습니다.
      '''
        GPU 메모리 부족으로 추측되는 오류가 발생하여 batch size를 32까지 줄이니 학습이 진행됐다.
        pre-trained 모델을 직접 만들어보니, 만들어진 것들이 진짜 잘 만들어진 모델이구나 싶다.
        아주 작은 모델인데도 에포크 돌리다가 세월 다 지나감.
        이번 노드에 대한 이해도가 떨어졌기 때문에 사실 프로젝트에 제출한 코드 수준은 그저 노드 옮겨적기였다.
        추가적인 학습을 통해 코드 한 줄 한 줄에 대한 이해가 필요할 것으로 보인다.
        특히, 하드 코딩이 아닌 함수화가 가능한 부분에 대해 함수화할 수 있도록 노력이 필요해보인다.
      '''
    
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 네. 대부분의 코드를 함수화해서 구현하였습니다.
      '''
        def build_model_pre_train(config):
            enc_tokens = tf.keras.layers.Input((None,), name="enc_tokens")
            segments = tf.keras.layers.Input((None,), name="segments")
        
            bert = BERT(config)
            logits_cls, logits_lm = bert((enc_tokens, segments))
        
            logits_cls = PooledOutput(config, 2, name="pooled_nsp")(logits_cls)
            outputs_nsp = tf.keras.layers.Softmax(name="nsp")(logits_cls)
        
            outputs_mlm = tf.keras.layers.Softmax(name="mlm")(logits_lm)
        
            model = tf.keras.Model(inputs=(enc_tokens, segments), outputs=(outputs_nsp, outputs_mlm))
            return model
      '''

# 참고 링크 및 코드 개선
```

```
