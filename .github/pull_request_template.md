# レビュイー欄／For reviewee

## 実装・実験・変更の概要／Summary of the implementation, experiment or changes 
<!--何のために何をどうしたかを簡潔に書く／Write simply what you did and its purpose.-->

- xxx

## コメント／Comments
<!--レビュアー向けの補足事項や別PRでやることを記載する.／Write supplements for the reviewer, or what to do in another PR.-->

- xxx

## 確認事項／Preparing

- [ ] Notebook以外の差分は500行以内に収まっていますか？また500行を超える場合はペアプロを実施しましたか？／Is the differences within 500 rows? If it isn't, did you do pair-programming with the reviwer in advance?
- [ ] AssigneesとReviewersを設定しましたか？／Have you set the assignees and reviewers?
- [ ] DSGのGitHub ProjectsにこのPRを紐づけましたか？（画面右側の`Projects`で`DS Developments`を紐づけてください）／Have you linked this PR to DSG's GitHub Projects?(Please link `DS Developments` in `Projects` on the right side navigation.)
- [ ] PRに対応するIssueを記載しましたか？（必要に応じて）／Have you linked the corresponding issue to this PR? (as needed.)
  - ref) [https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)


# レビュアー欄／For reviewer

## レビュー観点／Review perspectives

レビュアーはレビューした観点にチェックを入れてください。

Ref. [DSG Standard Coding Style](https://www.notion.so/abejainc/DSG-Standard-Coding-Style-now-Japanese-Only-626827849bcb4278b851d51623e49740?pvs=4)


- [ ] 変数・関数・クラスの役割が分かりやすいものになっているか／Whether the role of variables, functions and classes are easy to understand.
- [ ] 実装の背景がコメントとして書けているか、それを実装者以外の人が読んで理解出来るか／Whether or not the background of the implementation can be written as a comment, and whether or not a third party can read and understand it.
- [ ] docstringが書けているか／Whether the docstrings is prepared.
- [ ] 明らかなバグを見落としていないか／Whether any obvious bug could be included.
- [ ] 条件分岐やエラーハンドリングは適切か／Whether the conditional statement and error handling are appropriate.
- [ ] リークしていないか／Whether no leakage is found.
- [ ] バリデーション戦略または評価方法は適切か／Whether the validation strategy or the way to evaluate is appropriate.
- [ ] テストケースに漏れは無いか／Whether the representative test cases are covered well.
- [ ] 他の部分と書き方が異なっていないか／Whether the coding style is not different from the other parts of the codebase.
- [ ] より良い設計はないか／Whether there is an idea for better archetecture.
- [ ] 利用しているパッケージのライセンスに問題は無いか／Whether the license of used packages are fine for commercial use.
- [ ] クレデンシャルの情報がコミットに含まれていないか／Whether no credential is included in commits.
