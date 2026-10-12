(() => {
  const root = document.querySelector('[data-transformer-lab]');
  if (!root) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const flowStages = [
    {
      eyebrow: 'STEP 01 · DISCRETE INPUT',
      title: 'Tokenization gives the model indices, not meanings',
      description: 'Each token is an integer address into the embedding table. Geometry appears only after lookup.',
      tensor: 'token_ids: [T]',
      code: 'token_ids = tokenizer(text)\n# [T] integers in {0, …, |V| − 1}'
    },
    {
      eyebrow: 'STEP 02 · ENTER MODEL SPACE',
      title: 'Embedding lookup and position create the initial residual state',
      description: 'The lookup supplies token features; positional information breaks the symmetry between sequence locations.',
      tensor: 'X⁽⁰⁾: [T, d_model]',
      code: 'x = embedding[token_ids]\nx = x + positional_signal\n# x is the residual stream'
    },
    {
      eyebrow: 'STEP 03 · COMMUNICATE ACROSS POSITIONS',
      title: 'Masked self-attention gathers information from the prefix',
      description: 'Q and K determine where to read; V determines what is returned. The attention update is added back to the residual stream.',
      tensor: 'attention_update: [T, d_model]',
      code: 'h = norm(x)\nq, k, v = project_qkv(h)\na = softmax(q @ k.T / sqrt(d_head) + causal_mask)\nx = x + (a @ v) @ W_O'
    },
    {
      eyebrow: 'STEP 04 · COMPUTE WITHIN EACH POSITION',
      title: 'The MLP transforms features without mixing token positions',
      description: 'The same feed-forward network is applied independently to every row, and its update is added to the residual stream.',
      tensor: 'mlp_update: [T, d_model]',
      code: 'h = norm(x)\ngate = silu(h @ W_gate)\nupdate = (gate * (h @ W_up)) @ W_down\nx = x + update'
    },
    {
      eyebrow: 'STEP 05 · PREPARE THE READOUT',
      title: 'Final normalization controls the scale seen by the vocabulary head',
      description: 'After L blocks, the residual stream already contains the contextual representation for every prefix position.',
      tensor: 'H: [T, d_model]',
      code: 'h = final_norm(x)\n# h[t] depends only on token_ids[:t+1]'
    },
    {
      eyebrow: 'STEP 06 · RETURN TO VOCABULARY SPACE',
      title: 'Unembedding assigns a score to every possible next token',
      description: 'A learned linear map compares the contextual state with vocabulary directions and produces logits.',
      tensor: 'logits: [T, |V|]',
      code: 'logits = h @ W_U.T\n# one row of |V| scores per position'
    },
    {
      eyebrow: 'STEP 07 · NEXT-TOKEN DISTRIBUTION',
      title: 'Softmax turns the last-position logits into probabilities',
      description: 'During generation, one token is selected, appended to the sequence, and the forward pass continues with the longer prefix.',
      tensor: 'p(next token): [|V|]',
      code: 'probs = softmax(logits[-1], dim=-1)\nnext_id = sample(probs)\ntoken_ids = append(token_ids, next_id)'
    }
  ];

  const flowNodes = Array.from(root.querySelectorAll('[data-flow-stage]'));
  const flowPlay = root.querySelector('[data-flow-play]');
  const flowPrevious = root.querySelector('[data-flow-previous]');
  const flowNext = root.querySelector('[data-flow-next]');
  const flowProgress = root.querySelector('[data-flow-progress]');
  const flowEyebrow = root.querySelector('[data-flow-eyebrow]');
  const flowTitle = root.querySelector('[data-flow-title]');
  const flowDescription = root.querySelector('[data-flow-description]');
  const flowTensor = root.querySelector('[data-flow-tensor]');
  const flowCode = root.querySelector('[data-flow-code]');
  let flowIndex = 0;
  let flowTimer = null;

  const stopFlow = () => {
    if (flowTimer) window.clearInterval(flowTimer);
    flowTimer = null;
    flowPlay.textContent = 'Play forward pass';
    flowPlay.setAttribute('aria-pressed', 'false');
  };

  const renderFlow = () => {
    const stage = flowStages[flowIndex];
    flowNodes.forEach((node, index) => {
      const active = index === flowIndex;
      node.classList.toggle('is-active', active);
      node.classList.toggle('is-complete', index < flowIndex);
      node.setAttribute('aria-pressed', String(active));
    });
    flowProgress.textContent = `${flowIndex + 1} / ${flowStages.length}`;
    flowEyebrow.textContent = stage.eyebrow;
    flowTitle.textContent = stage.title;
    flowDescription.textContent = stage.description;
    flowTensor.textContent = stage.tensor;
    flowCode.textContent = stage.code;
    flowPrevious.disabled = flowIndex === 0;
    flowNext.disabled = flowIndex === flowStages.length - 1;
  };

  flowNodes.forEach((node) => {
    node.addEventListener('click', () => {
      stopFlow();
      flowIndex = Number(node.dataset.flowStage);
      renderFlow();
    });
  });
  flowPrevious.addEventListener('click', () => {
    stopFlow();
    flowIndex = Math.max(0, flowIndex - 1);
    renderFlow();
  });
  flowNext.addEventListener('click', () => {
    stopFlow();
    flowIndex = Math.min(flowStages.length - 1, flowIndex + 1);
    renderFlow();
  });
  flowPlay.addEventListener('click', () => {
    if (flowTimer) {
      stopFlow();
      return;
    }
    if (flowIndex === flowStages.length - 1) flowIndex = 0;
    renderFlow();
    flowPlay.textContent = 'Pause';
    flowPlay.setAttribute('aria-pressed', 'true');
    flowTimer = window.setInterval(() => {
      if (flowIndex === flowStages.length - 1) {
        stopFlow();
        return;
      }
      flowIndex += 1;
      renderFlow();
    }, reducedMotion ? 2200 : 1350);
  });

  const tokens = ['The', 'cat', 'did', 'not', 'sleep', 'yet'];
  const queries = [
    [1.00, 0.10], [0.90, 0.30], [0.20, 0.90],
    [0.10, 1.10], [0.90, 0.70], [0.30, 0.80]
  ];
  const keys = [
    [0.90, 0.10], [1.00, 0.40], [0.20, 0.80],
    [0.00, 1.20], [0.70, 0.80], [0.30, 0.90]
  ];
  const values = [
    [1.00, 0.10], [0.80, 0.50], [0.20, 0.80],
    [-0.70, 1.00], [0.50, 0.90], [0.00, 0.70]
  ];

  const tokenGroup = root.querySelector('[data-query-tokens]');
  const causalInput = root.querySelector('[data-causal]');
  const attentionRows = root.querySelector('[data-attention-rows]');
  const attentionMatrix = root.querySelector('[data-attention-matrix]');
  const attentionButtons = Array.from(root.querySelectorAll('[data-attention-stage]'));
  const attentionPlay = root.querySelector('[data-attention-play]');
  const queryLabel = root.querySelector('[data-query-label]');
  const queryVector = root.querySelector('[data-query-vector]');
  const attentionOutput = root.querySelector('[data-attention-output]');
  const attentionEyebrow = root.querySelector('[data-attention-eyebrow]');
  const attentionFormula = root.querySelector('[data-attention-formula]');
  const attentionCode = root.querySelector('[data-attention-code]');
  let queryIndex = 4;
  let attentionStage = 0;
  let attentionTimer = null;

  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  const format = (value) => Number.isFinite(value) ? value.toFixed(2) : '−∞';
  const softmax = (scores) => {
    const finite = scores.filter(Number.isFinite);
    const max = Math.max(...finite);
    const exponentials = scores.map((score) => Number.isFinite(score) ? Math.exp(score - max) : 0);
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
  };

  const calculate = (rowIndex, causal) => {
    const raw = keys.map((key) => dot(queries[rowIndex], key));
    const scaled = raw.map((score) => score / Math.sqrt(queries[rowIndex].length));
    const masked = scaled.map((score, keyIndex) => causal && keyIndex > rowIndex ? -Infinity : score);
    const weights = softmax(masked);
    const output = values[0].map((_, dimension) => (
      weights.reduce((sum, weight, index) => sum + weight * values[index][dimension], 0)
    ));
    return {raw, scaled, masked, weights, output};
  };

  const attentionStages = [
    {
      eyebrow: 'STEP 01 · LINEAR PROJECTIONS',
      formula: 'qᵢ = WQ xᵢ · kⱼ = WK xⱼ · vⱼ = WV xⱼ',
      code: 'Q = X @ W_Q\nK = X @ W_K\nV = X @ W_V'
    },
    {
      eyebrow: 'STEP 02 · COMPATIBILITY',
      formula: 'sᵢⱼ = qᵢᵀkⱼ',
      code: 'scores = Q @ K.transpose(-2, -1)\n# scores[i, j] compares query i with key j'
    },
    {
      eyebrow: 'STEP 03 · SCALE AND MASK',
      formula: 's̃ᵢⱼ = sᵢⱼ / √dₖ + Mᵢⱼ',
      code: 'scores = scores / math.sqrt(d_head)\nscores = scores.masked_fill(causal_mask, -inf)'
    },
    {
      eyebrow: 'STEP 04 · NORMALIZE',
      formula: 'aᵢⱼ = exp(s̃ᵢⱼ) / Σᵤ exp(s̃ᵢᵤ)',
      code: 'weights = torch.softmax(scores, dim=-1)\n# every query row sums to one'
    },
    {
      eyebrow: 'STEP 05 · RETRIEVE VALUES',
      formula: 'yᵢ = Σⱼ aᵢⱼvⱼ',
      code: 'context = weights @ V\n# context[i] is a weighted value average'
    }
  ];

  const stopAttention = () => {
    if (attentionTimer) window.clearInterval(attentionTimer);
    attentionTimer = null;
    attentionPlay.textContent = 'Play calculation';
    attentionPlay.setAttribute('aria-pressed', 'false');
  };

  const renderTokenButtons = () => {
    tokenGroup.innerHTML = '';
    tokens.forEach((token, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = token;
      button.className = index === queryIndex ? 'is-active' : '';
      button.setAttribute('aria-pressed', String(index === queryIndex));
      button.addEventListener('click', () => {
        stopAttention();
        queryIndex = index;
        renderAttention();
      });
      tokenGroup.append(button);
    });
  };

  const renderRows = (calculation) => {
    const maximum = Math.max(...calculation.weights);
    attentionRows.innerHTML = '';
    tokens.forEach((token, index) => {
      const row = document.createElement('div');
      row.className = 'attention-vector-row';
      if (!Number.isFinite(calculation.masked[index])) row.classList.add('is-masked');
      const visibleMetric = attentionStage === 0
        ? `k=[${keys[index].map(format).join(', ')}] · v=[${values[index].map(format).join(', ')}]`
        : attentionStage === 1
          ? `q·k = ${format(calculation.raw[index])}`
          : attentionStage === 2
            ? `scaled = ${format(calculation.masked[index])}`
            : `weight = ${format(calculation.weights[index])}`;
      const width = attentionStage < 3
        ? Math.max(4, Math.abs(calculation.scaled[index]) / 1.3 * 100)
        : calculation.weights[index] / maximum * 100;
      row.innerHTML = `
        <span class="attention-vector-row__token">${token}</span>
        <span class="attention-vector-row__bar"><i style="--attention-width:${Math.min(100, width)}%"></i></span>
        <code>${visibleMetric}</code>
      `;
      attentionRows.append(row);
    });
  };

  const renderMatrix = () => {
    attentionMatrix.innerHTML = '';
    const corner = document.createElement('span');
    corner.className = 'attention-matrix__corner';
    corner.textContent = 'q↓ k→';
    attentionMatrix.append(corner);
    tokens.forEach((token) => {
      const label = document.createElement('span');
      label.className = 'attention-matrix__label';
      label.textContent = token;
      attentionMatrix.append(label);
    });
    tokens.forEach((queryToken, rowIndex) => {
      const rowLabel = document.createElement('span');
      rowLabel.className = 'attention-matrix__label attention-matrix__label--row';
      rowLabel.textContent = queryToken;
      attentionMatrix.append(rowLabel);
      const row = calculate(rowIndex, causalInput.checked);
      row.weights.forEach((weight, columnIndex) => {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'attention-matrix__cell';
        if (rowIndex === queryIndex) cell.classList.add('is-selected-row');
        if (causalInput.checked && columnIndex > rowIndex) cell.classList.add('is-masked');
        cell.style.setProperty('--weight', weight.toFixed(3));
        cell.textContent = attentionStage >= 3 ? weight.toFixed(2) : '·';
        cell.title = `${queryToken} attends to ${tokens[columnIndex]} with weight ${weight.toFixed(3)}`;
        cell.setAttribute('aria-label', cell.title);
        cell.addEventListener('click', () => {
          stopAttention();
          queryIndex = rowIndex;
          attentionStage = 3;
          renderAttention();
        });
        attentionMatrix.append(cell);
      });
    });
  };

  const renderAttention = () => {
    const calculation = calculate(queryIndex, causalInput.checked);
    const stage = attentionStages[attentionStage];
    renderTokenButtons();
    renderRows(calculation);
    renderMatrix();
    queryLabel.textContent = tokens[queryIndex];
    queryVector.textContent = `q = [${queries[queryIndex].map(format).join(', ')}]`;
    attentionButtons.forEach((button, index) => {
      const active = index === attentionStage;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    attentionEyebrow.textContent = stage.eyebrow;
    attentionFormula.textContent = stage.formula;
    attentionCode.textContent = stage.code;

    if (attentionStage === 0) {
      attentionOutput.textContent = `The selected token “${tokens[queryIndex]}” contributes one query. Every token contributes a key and a value.`;
    } else if (attentionStage === 1) {
      attentionOutput.textContent = `The query for “${tokens[queryIndex]}” is compared with all ${tokens.length} keys by a dot product.`;
    } else if (attentionStage === 2) {
      attentionOutput.textContent = causalInput.checked
        ? `Keys after “${tokens[queryIndex]}” are replaced by −∞, so their later softmax weights must be zero.`
        : 'The causal mask is off, so this query can read both earlier and later positions.';
    } else if (attentionStage === 3) {
      const strongest = calculation.weights.indexOf(Math.max(...calculation.weights));
      attentionOutput.textContent = `The weights sum to 1.00; “${tokens[strongest]}” receives the largest weight for this toy query.`;
    } else {
      attentionOutput.textContent = `Weighted value sum for “${tokens[queryIndex]}”: y = [${calculation.output.map(format).join(', ')}].`;
    }
  };

  attentionButtons.forEach((button) => {
    button.addEventListener('click', () => {
      stopAttention();
      attentionStage = Number(button.dataset.attentionStage);
      renderAttention();
    });
  });
  causalInput.addEventListener('change', () => {
    stopAttention();
    renderAttention();
  });
  attentionPlay.addEventListener('click', () => {
    if (attentionTimer) {
      stopAttention();
      return;
    }
    if (attentionStage === attentionStages.length - 1) attentionStage = 0;
    renderAttention();
    attentionPlay.textContent = 'Pause';
    attentionPlay.setAttribute('aria-pressed', 'true');
    attentionTimer = window.setInterval(() => {
      if (attentionStage === attentionStages.length - 1) {
        stopAttention();
        return;
      }
      attentionStage += 1;
      renderAttention();
    }, reducedMotion ? 2200 : 1450);
  });

  const loadExternalButton = root.querySelector('[data-load-external]');
  const externalFrame = root.querySelector('[data-external-frame]');
  loadExternalButton.addEventListener('click', () => {
    const iframe = document.createElement('iframe');
    iframe.src = 'https://poloclub.github.io/transformer-explainer/';
    iframe.title = 'Transformer Explainer by Georgia Tech Polo Club';
    iframe.loading = 'lazy';
    iframe.referrerPolicy = 'no-referrer';
    iframe.allow = 'fullscreen; clipboard-read; clipboard-write';
    iframe.sandbox = 'allow-scripts allow-same-origin allow-forms allow-popups allow-downloads';
    externalFrame.innerHTML = '';
    externalFrame.append(iframe);
  });

  renderFlow();
  renderAttention();
})();
