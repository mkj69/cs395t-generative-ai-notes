(() => {
  const root = document.querySelector('[data-rlvr-lab]');
  if (!root) return;

  const responseElements = Array.from(root.querySelectorAll('[data-response]'));
  const rewardButtons = Array.from(root.querySelectorAll('[data-reward-toggle]'));
  const algorithmButtons = Array.from(root.querySelectorAll('[data-algorithm]'));
  const rewardVector = root.querySelector('[data-reward-vector]');
  const groupMean = root.querySelector('[data-group-mean]');
  const normalizer = root.querySelector('[data-normalizer]');
  const algorithmName = root.querySelector('[data-algorithm-name]');
  const formula = root.querySelector('[data-formula]');
  const explanation = root.querySelector('[data-explanation]');
  const codeSnippet = root.querySelector('[data-code-snippet]');
  const replayButton = root.querySelector('[data-replay]');
  const nextScenarioButton = root.querySelector('[data-next-scenario]');
  const trainingTrace = root.querySelector('[data-training-trace]');
  const traceStageButtons = Array.from(trainingTrace.querySelectorAll('[data-trace-stage]'));
  const traceEyebrow = trainingTrace.querySelector('[data-stage-eyebrow]');
  const traceTitle = trainingTrace.querySelector('[data-stage-title]');
  const traceDescription = trainingTrace.querySelector('[data-stage-description]');
  const traceInput = trainingTrace.querySelector('[data-stage-input]');
  const traceOperation = trainingTrace.querySelector('[data-stage-operation]');
  const traceOutput = trainingTrace.querySelector('[data-stage-output]');
  const traceAlgorithm = trainingTrace.querySelector('[data-stage-algorithm]');
  const traceNote = trainingTrace.querySelector('[data-stage-note]');
  const traceShape = trainingTrace.querySelector('[data-stage-shape]');
  const traceCode = trainingTrace.querySelector('[data-stage-code]');
  const tokenLossView = trainingTrace.querySelector('[data-token-loss-view]');
  const tokenRows = trainingTrace.querySelector('[data-token-rows]');
  const previousStageButton = trainingTrace.querySelector('[data-stage-previous]');
  const nextStageButton = trainingTrace.querySelector('[data-stage-next]');
  const stageProgress = trainingTrace.querySelector('[data-stage-progress]');

  const lengths = [4, 8, 12, 16];
  const scenarios = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0]
  ];
  let rewards = scenarios[0].slice();
  let scenarioIndex = 0;
  let activeAlgorithm = 'rloo';
  let activeStage = 0;

  const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

  const algorithms = {
    reinforce: {
      name: 'REINFORCE',
      formula: 'Rᵢ',
      description: 'Only successful responses receive a positive score weight. Failed responses contribute zero before a baseline is introduced.',
      code: [
        '# response_logp: [G], each entry sums token log-probabilities',
        'advantage = rewards',
        'loss = -(advantage.detach() * response_logp).mean()'
      ].join('\n'),
      compute: () => ({
        coefficients: rewards.slice(),
        normalizer: 'none'
      })
    },
    rloo: {
      name: 'RLOO',
      formula: 'Rᵢ − mean(Rⱼ, j ≠ i)',
      description: 'The current response is excluded from its own baseline, so the estimator remains unbiased for the expected-reward gradient.',
      code: [
        'G = rewards.numel()',
        'loo_baseline = (rewards.sum() - rewards) / (G - 1)',
        'advantage = rewards - loo_baseline',
        'loss = -(advantage.detach() * response_logp).mean()'
      ].join('\n'),
      compute: () => {
        const total = rewards.reduce((sum, value) => sum + value, 0);
        return {
          coefficients: rewards.map((reward) => reward - (total - reward) / (rewards.length - 1)),
          normalizer: 'leave-one-out'
        };
      }
    },
    drgrpo: {
      name: 'Dr. GRPO',
      formula: 'Rᵢ − R̄',
      description: 'Including the response in the group mean scales RLOO by (G−1)/G. For a fixed group size, the direction is unchanged.',
      code: [
        'group_mean = rewards.mean()',
        'advantage = rewards - group_mean',
        'loss = -(advantage.detach() * response_logp).mean()'
      ].join('\n'),
      compute: () => {
        const groupAverage = mean(rewards);
        return {
          coefficients: rewards.map((reward) => reward - groupAverage),
          normalizer: 'center only'
        };
      }
    },
    grpo: {
      name: 'GRPO',
      formula: '(Rᵢ − R̄) / (Tᵢ · √(R̄(1−R̄)))',
      description: 'Standard-deviation normalization reweights prompt difficulty; dividing by response length also changes the relative weight of short and long trajectories.',
      code: [
        'group_mean = rewards.mean()',
        'group_std = rewards.std(unbiased=False)',
        'advantage = torch.where(',
        '    group_std > 0,',
        '    (rewards - group_mean) / group_std,',
        '    torch.zeros_like(rewards)',
        ')',
        'per_response = advantage.detach() * token_logp.sum(-1) / lengths',
        'loss = -per_response.mean() + beta * sampled_kl.mean()'
      ].join('\n'),
      compute: () => {
        const groupAverage = mean(rewards);
        const std = Math.sqrt(groupAverage * (1 - groupAverage));
        return {
          coefficients: std === 0
            ? rewards.map(() => 0)
            : rewards.map((reward, index) => (reward - groupAverage) / (std * lengths[index])),
          normalizer: std === 0 ? 'zero: rewards agree' : 'std × length'
        };
      }
    },
    maxrl: {
      name: 'Practical MaxRL',
      formula: '(Rᵢ − R̄) / R̄',
      description: 'Dividing by the observed success rate amplifies hard groups. All-failure groups are skipped, which makes the practical update target order G−1.',
      code: [
        'group_mean = rewards.mean()',
        'if group_mean == 0:',
        '    return None  # skip the all-failure group',
        'advantage = (rewards - group_mean) / group_mean',
        'loss = -(advantage.detach() * response_logp).mean()'
      ].join('\n'),
      compute: () => {
        const groupAverage = mean(rewards);
        return {
          coefficients: groupAverage === 0
            ? rewards.map(() => 0)
            : rewards.map((reward) => (reward - groupAverage) / groupAverage),
          normalizer: groupAverage === 0 ? 'skip all-failure group' : 'divide by R̄'
        };
      }
    }
  };

  const formatCoefficient = (value) => {
    if (Math.abs(value) < 0.0005) return '0.000';
    return (value > 0 ? '+' : '') + value.toFixed(3);
  };

  const advantageCode = {
    reinforce: [
      'advantage = rewards                         # [G]'
    ],
    rloo: [
      'reward_sum = rewards.sum(dim=-1, keepdim=True)',
      'baseline = (reward_sum - rewards) / (G - 1)',
      'advantage = rewards - baseline              # [G]'
    ],
    drgrpo: [
      'group_mean = rewards.mean(dim=-1, keepdim=True)',
      'advantage = rewards - group_mean            # [G]'
    ],
    grpo: [
      'group_mean = rewards.mean(dim=-1, keepdim=True)',
      'group_std = rewards.std(dim=-1, unbiased=False, keepdim=True)',
      'advantage = torch.where(',
      '    group_std > 0,',
      '    (rewards - group_mean) / group_std,',
      '    torch.zeros_like(rewards),',
      ')                                             # [G]'
    ],
    maxrl: [
      'group_mean = rewards.mean(dim=-1, keepdim=True)',
      'valid_group = group_mean > 0',
      'advantage = torch.where(',
      '    valid_group,',
      '    (rewards - group_mean) / group_mean.clamp_min(1e-8),',
      '    torch.zeros_like(rewards),',
      ')                                             # [G]'
    ]
  };

  const advantageNotes = {
    reinforce: 'No baseline is subtracted. A failed response therefore has zero reward weight in the raw estimator.',
    rloo: 'Response i never enters its own baseline. Conditional independence makes the baseline term zero in expectation.',
    drgrpo: 'The response enters the group mean, scaling RLOO by (G−1)/G while preserving its direction for fixed G.',
    grpo: 'Centering is followed by group-standard-deviation normalization. The response-length factor is applied in the token-loss stage.',
    maxrl: 'The centered reward is divided by the observed success rate. An all-failure group is marked invalid and skipped.'
  };

  const regularizationStages = {
    reinforce: {
      operation: 'on-policy score loss',
      output: 'no ratio needed',
      note: 'Pure REINFORCE uses a freshly sampled batch once. PPO clipping is an optional training wrapper, not part of the estimator itself.',
      code: [
        '# Fresh on-policy data: πθ is the sampling policy',
        'ratio = 1.0',
        'regularized_loss = policy_loss',
        '# entropy or KL terms may be added by the trainer'
      ]
    },
    rloo: {
      operation: 'baseline only',
      output: 'unclipped RLOO loss',
      note: 'RLOO defines the baseline. It can be used on-policy as shown here or placed inside a separate PPO-style optimization loop.',
      code: [
        '# RLOO changes the advantage, not the optimizer wrapper',
        'regularized_loss = policy_loss',
        '# If batches are reused, compute old_logp ratios explicitly'
      ]
    },
    drgrpo: {
      operation: 'PPO-style sample reuse',
      output: 'clipped token objective',
      note: 'Practical Dr. GRPO often freezes the centered advantages and reuses the batch with token-level PPO clipping.',
      code: [
        'ratio = torch.exp(new_logp - old_logp.detach())',
        'unclipped = ratio * advantage[:, None]',
        'clipped = ratio.clamp(1 - eps, 1 + eps) * advantage[:, None]',
        'policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask.sum()'
      ]
    },
    grpo: {
      operation: 'clip ratios + reference KL',
      output: 'policy loss + β KL',
      note: 'π_old generated the batch; π_ref supplies the regularization anchor. These policies have different roles.',
      code: [
        'ratio = torch.exp(new_logp - old_logp.detach())',
        'surrogate = torch.minimum(',
        '    ratio * advantage[:, None],',
        '    ratio.clamp(1 - eps, 1 + eps) * advantage[:, None],',
        ')',
        'u = torch.exp(ref_logp - new_logp)',
        'sampled_kl = u - torch.log(u) - 1',
        'regularized_loss = reward_loss + beta * masked_mean(sampled_kl, mask)'
      ]
    },
    maxrl: {
      operation: 'choose optimizer wrapper',
      output: 'MaxRL-weighted loss',
      note: 'MaxRL specifies difficulty weighting and the skip rule. PPO clipping or KL regularization may be added, but neither defines MaxRL.',
      code: [
        '# MaxRL supplies `advantage`; the wrapper is a separate choice',
        'if reuse_old_batch:',
        '    policy_loss = ppo_surrogate(new_logp, old_logp, advantage, mask)',
        'else:',
        '    policy_loss = reinforce_loss(new_logp, advantage, mask)'
      ]
    }
  };

  const getTraceStages = () => {
    const algorithm = algorithms[activeAlgorithm];
    const result = algorithm.compute();
    const coefficientVector = '[' + result.coefficients.map(formatCoefficient).join(', ') + ']';
    const tokenWeightCode = activeAlgorithm === 'grpo'
      ? [
          'lengths = mask.sum(dim=-1)                       # [G]',
          'token_weight = advantage[:, None] / lengths[:, None]',
          'reward_loss = -(token_weight.detach() * token_logp * mask).sum() / G'
        ]
      : [
          'token_weight = advantage[:, None]                # [G, 1]',
          'response_score = (token_logp * mask).sum(dim=-1) # [G]',
          'reward_loss = -(advantage.detach() * response_score).mean()'
        ];
    const regularization = regularizationStages[activeAlgorithm];

    return [
      {
        eyebrow: 'STEP 01 · DATA COLLECTION',
        title: 'Sample several continuations from a frozen policy',
        description: 'The prompt is repeated G times. Sampling uses π_old, which stays fixed while this batch is reused.',
        input: 'prompt_ids [B, L]',
        operation: 'sample G responses',
        output: 'tokens, mask, old_logp [B, G, T]',
        note: 'The estimator has not acted yet; every method begins from the same sampled group.',
        shape: 'G responses per prompt',
        code: [
          'with torch.no_grad():',
          '    tokens = policy_old.generate(',
          '        prompt_ids, num_return_sequences=G',
          '    )',
          '    old_logp = policy_old.log_probs(tokens)',
          'mask = tokens.ne(pad_token_id)'
        ],
        showTokens: false
      },
      {
        eyebrow: 'STEP 02 · VERIFIABLE REWARD',
        title: 'Score complete responses outside the model',
        description: 'The verifier consumes each completed response and returns one terminal reward. No gradient passes through this operation.',
        input: 'decoded responses [G]',
        operation: 'verifier(x, yᵢ)',
        output: 'rewards = [' + rewards.join(', ') + ']',
        note: 'All five methods see the same reward vector. Their differences begin when this vector is converted into an advantage.',
        shape: 'one scalar per response',
        code: [
          'with torch.no_grad():',
          '    rewards = verifier(prompt, decoded_responses)',
          '    rewards = rewards.float()                    # [G]',
          '# rewards are data; do not differentiate through the verifier'
        ],
        showTokens: false
      },
      {
        eyebrow: 'STEP 03 · CREDIT ASSIGNMENT',
        title: 'Turn rewards into response-level advantages',
        description: algorithm.formula + ' produces the scalar that will multiply each response score.',
        input: 'rewards [' + rewards.join(', ') + ']',
        operation: algorithm.formula,
        output: 'advantage = ' + coefficientVector,
        note: advantageNotes[activeAlgorithm],
        shape: 'advantage [G]',
        code: advantageCode[activeAlgorithm],
        showTokens: false
      },
      {
        eyebrow: 'STEP 04 · TOKEN-LEVEL LOSS',
        title: 'Broadcast each response coefficient across its tokens',
        description: 'The response log-probability is a sum of token log-probabilities. Padding is masked, and only GRPO divides the reward contribution by response length in this comparison.',
        input: 'advantage [G] + token_logp [G, T]',
        operation: activeAlgorithm === 'grpo' ? 'broadcast, mask, divide by Tᵢ' : 'broadcast, mask, sum tokens',
        output: 'scalar reward_loss',
        note: activeAlgorithm === 'grpo'
          ? 'Long responses receive a smaller per-token coefficient because the response contribution is averaged over valid tokens.'
          : 'Every valid token in one response receives the same response-level coefficient before its own score vector is applied.',
        shape: 'token_weight [G, T]',
        code: [
          'token_logp = policy.log_probs(tokens)             # [G, T]',
          ...tokenWeightCode
        ],
        showTokens: true
      },
      {
        eyebrow: 'STEP 05 · OPTIMIZATION WRAPPER',
        title: 'Decide how far the policy may move on this batch',
        description: 'Clipping and reference-policy regularization are separate from the baseline calculation. The exact wrapper depends on the algorithm and implementation.',
        input: 'new_logp, old_logp, advantage',
        operation: regularization.operation,
        output: regularization.output,
        note: regularization.note,
        shape: activeAlgorithm === 'grpo' || activeAlgorithm === 'drgrpo' ? 'token ratios [G, T]' : 'trainer-dependent',
        code: regularization.code,
        showTokens: false
      },
      {
        eyebrow: 'STEP 06 · PARAMETER UPDATE',
        title: 'Combine every token contribution and update θ',
        description: 'Autograd adds the weighted token score vectors. Positive coefficients locally raise sampled-token probability; negative coefficients lower it.',
        input: 'regularized scalar loss',
        operation: 'backward + optimizer.step',
        output: 'θ ← θ − η∇θ loss',
        note: 'The animation shows scalar weights, not the full parameter gradient. Two responses can partially cancel because their score vectors point in different directions.',
        shape: 'millions or billions of parameter gradients',
        code: [
          'optimizer.zero_grad(set_to_none=True)',
          'regularized_loss.backward()',
          'torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)',
          'optimizer.step()',
          '# resample after the configured number of reuse steps'
        ],
        showTokens: false
      }
    ];
  };

  const renderTokenRows = (coefficients) => {
    tokenRows.replaceChildren();
    coefficients.forEach((coefficient, index) => {
      const row = document.createElement('div');
      row.className = 'token-loss-row';

      const label = document.createElement('span');
      label.textContent = 'y' + ['¹', '²', '³', '⁴'][index] + ' · T=' + lengths[index];

      const tokens = document.createElement('div');
      tokens.className = 'token-loss-row__tokens';
      tokens.setAttribute('aria-hidden', 'true');
      for (let tokenIndex = 0; tokenIndex < lengths[index]; tokenIndex += 1) {
        const token = document.createElement('i');
        if (coefficient > 0.0005) token.className = 'is-positive';
        if (coefficient < -0.0005) token.className = 'is-negative';
        tokens.appendChild(token);
      }

      const output = document.createElement('output');
      output.textContent = formatCoefficient(coefficient) + ' / token';
      row.append(label, tokens, output);
      tokenRows.appendChild(row);
    });
  };

  const renderTrace = () => {
    const stages = getTraceStages();
    const stage = stages[activeStage];
    const algorithm = algorithms[activeAlgorithm];

    traceEyebrow.textContent = stage.eyebrow;
    traceTitle.textContent = stage.title;
    traceDescription.textContent = stage.description;
    traceInput.textContent = stage.input;
    traceOperation.textContent = stage.operation;
    traceOutput.textContent = stage.output;
    traceAlgorithm.textContent = algorithm.name;
    traceNote.textContent = stage.note;
    traceShape.textContent = stage.shape;
    traceCode.textContent = stage.code.join('\n');
    tokenLossView.hidden = !stage.showTokens;
    stageProgress.textContent = (activeStage + 1) + ' / ' + stages.length;
    previousStageButton.disabled = activeStage === 0;
    nextStageButton.disabled = activeStage === stages.length - 1;

    traceStageButtons.forEach((button, index) => {
      const isActive = index === activeStage;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });

    if (stage.showTokens) renderTokenRows(algorithm.compute().coefficients);
  };

  const render = () => {
    const groupAverage = mean(rewards);
    const algorithm = algorithms[activeAlgorithm];
    const result = algorithm.compute();
    const maxMagnitude = Math.max(0.0001, ...result.coefficients.map((value) => Math.abs(value)));

    rewardVector.textContent = rewards.join(', ');
    groupMean.textContent = groupAverage.toFixed(2);
    normalizer.textContent = result.normalizer;
    algorithmName.textContent = algorithm.name;
    formula.textContent = algorithm.formula;
    explanation.textContent = algorithm.description;
    codeSnippet.textContent = algorithm.code;

    algorithmButtons.forEach((button) => {
      const isActive = button.dataset.algorithm === activeAlgorithm;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });

    responseElements.forEach((element, index) => {
      const reward = rewards[index];
      const coefficient = result.coefficients[index];
      const rewardButton = element.querySelector('[data-reward-toggle]');
      const rewardOutput = element.querySelector('[data-reward]');
      const coefficientOutput = element.querySelector('[data-coefficient]');
      const coefficientFill = element.querySelector('[data-coefficient-fill]');
      const coefficientTrack = element.querySelector('[data-coefficient-track]');
      const width = Math.abs(coefficient) / maxMagnitude * 48;
      const isNegative = coefficient < -0.0005;

      rewardOutput.textContent = String(reward);
      rewardButton.classList.toggle('is-success', reward === 1);
      rewardButton.setAttribute('aria-pressed', String(reward === 1));
      rewardButton.setAttribute(
        'aria-label',
        'Response ' + (index + 1) + ' verifier reward: ' + (reward === 1 ? 'success' : 'failure')
      );

      coefficientOutput.textContent = formatCoefficient(coefficient);
      coefficientFill.classList.toggle('is-negative', isNegative);
      coefficientFill.style.width = width.toFixed(2) + '%';
      coefficientFill.style.left = (coefficient < 0 ? 50 - width : 50).toFixed(2) + '%';
      coefficientTrack.setAttribute(
        'aria-label',
        'Response ' + (index + 1) + ' score coefficient ' + formatCoefficient(coefficient)
      );
    });

    renderTrace();
  };

  rewardButtons.forEach((button, index) => {
    button.addEventListener('click', () => {
      rewards[index] = rewards[index] === 1 ? 0 : 1;
      render();
    });
  });

  algorithmButtons.forEach((button) => {
    button.addEventListener('click', () => {
      activeAlgorithm = button.dataset.algorithm;
      render();
    });
  });

  replayButton.addEventListener('click', () => {
    root.classList.remove('is-replaying');
    void root.offsetWidth;
    root.classList.add('is-replaying');
    window.setTimeout(() => root.classList.remove('is-replaying'), 650);
  });

  nextScenarioButton.addEventListener('click', () => {
    scenarioIndex = (scenarioIndex + 1) % scenarios.length;
    rewards = scenarios[scenarioIndex].slice();
    render();
  });

  traceStageButtons.forEach((button) => {
    button.addEventListener('click', () => {
      activeStage = Number(button.dataset.traceStage);
      renderTrace();
    });
  });

  previousStageButton.addEventListener('click', () => {
    activeStage = Math.max(0, activeStage - 1);
    renderTrace();
  });

  nextStageButton.addEventListener('click', () => {
    activeStage = Math.min(traceStageButtons.length - 1, activeStage + 1);
    renderTrace();
  });

  const ppo = root.querySelector('[data-ppo-explorer]');
  const ratioInput = ppo.querySelector('[data-ratio]');
  const epsilonInput = ppo.querySelector('[data-epsilon]');
  const ratioOutput = ppo.querySelector('[data-ratio-output]');
  const epsilonOutput = ppo.querySelector('[data-epsilon-output]');
  const advantageButtons = Array.from(ppo.querySelectorAll('[data-advantage]'));
  const clipBand = ppo.querySelector('[data-clip-band]');
  const ratioMarker = ppo.querySelector('[data-ratio-marker]');
  const ppoTrack = ppo.querySelector('[data-ppo-track]');
  const unclippedOutput = ppo.querySelector('[data-unclipped]');
  const objectiveOutput = ppo.querySelector('[data-ppo-objective]');
  const clipStatus = ppo.querySelector('[data-clip-status]');
  let advantage = 1;

  const renderPpo = () => {
    const ratio = Number(ratioInput.value);
    const epsilon = Number(epsilonInput.value);
    const lower = 1 - epsilon;
    const upper = 1 + epsilon;
    const clippedRatio = Math.min(upper, Math.max(lower, ratio));
    const unclipped = ratio * advantage;
    const clippedCandidate = clippedRatio * advantage;
    const objective = Math.min(unclipped, clippedCandidate);
    const isClipped = advantage > 0 ? ratio > upper : ratio < lower;
    const ratioPosition = (ratio - 0.5) * 100;
    const bandLeft = (lower - 0.5) * 100;
    const bandWidth = (upper - lower) * 100;

    ratioOutput.textContent = ratio.toFixed(2);
    epsilonOutput.textContent = epsilon.toFixed(2);
    unclippedOutput.textContent = unclipped.toFixed(3);
    objectiveOutput.textContent = objective.toFixed(3);
    clipStatus.textContent = isClipped ? 'clipped' : 'active';
    ratioMarker.style.left = ratioPosition.toFixed(1) + '%';
    ratioMarker.classList.toggle('is-clipped', isClipped);
    clipBand.style.left = bandLeft.toFixed(1) + '%';
    clipBand.style.width = bandWidth.toFixed(1) + '%';
    ppoTrack.setAttribute(
      'aria-label',
      'PPO ratio ' + ratio.toFixed(2) + ', clipping interval ' + lower.toFixed(2) + ' to ' + upper.toFixed(2) +
      ', advantage ' + (advantage > 0 ? 'positive' : 'negative') + ', gradient ' + (isClipped ? 'clipped' : 'active')
    );

    advantageButtons.forEach((button) => {
      const isActive = Number(button.dataset.advantage) === advantage;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });
  };

  ratioInput.addEventListener('input', renderPpo);
  epsilonInput.addEventListener('input', renderPpo);
  advantageButtons.forEach((button) => {
    button.addEventListener('click', () => {
      advantage = Number(button.dataset.advantage);
      renderPpo();
    });
  });

  render();
  renderPpo();
})();
