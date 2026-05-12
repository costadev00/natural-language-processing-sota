# Politica de Artefatos

## O que Fica no Git

- Codigo-fonte, notebooks didaticos e configuracoes pequenas.
- READMEs, manifests `project.yaml`, relatorios Markdown/LaTeX e bibliografia.
- Metricas agregadas pequenas em JSON/CSV.
- PDFs de papers e relatorios de disciplina.

## O que Fica Fora do Git

- Datasets locais em Arrow ou diretorios `DatasetDict`.
- Checkpoints, pesos de modelos, otimizadores e diretorios `model/`.
- Ambientes virtuais, caches, logs e staging de upload.
- Dumps grandes de predicoes ou geracoes em JSONL.

Esses artefatos sao ignorados por `.gitignore`. Quando necessario, eles podem ser reconstruidos pelos scripts da trilha ou baixados de fontes externas documentadas.

## RLHF

O dataset preparado da trilha RLHF esta publicado no Hugging Face Hub:

```text
costadev00/dolly-15k-rlhf-instructgpt-format
```

O config `rm_synthetic` usa uma preferencia proxy: a resposta Dolly e tratada como `chosen`, e uma resposta gerada pelo modelo SFT e tratada como `rejected`. Isso permite exercitar reward modeling, mas nao deve ser interpretado como dado de preferencia humana.

## Historico Git

A reorganizacao remove artefatos pesados da arvore versionada atual, mas nao reescreve o historico do Git. Uma limpeza historica com `git-filter-repo` ou BFG pode ser feita no futuro, com backup e coordenacao dos clones.
