## Dirichlet Process (DP)
DPはDirichlet分布のNon-parametric Bayesianな拡張のこと．
とっかかりとなるのはDirichlet分布のAggregation Propertyで，
これはDirichlet DistributedなN次元変数
`$$(X_1, \cdots, X_N) \sim \mathrm{Dir}(\alpha_1, \cdots, \alpha_N)$$`
がある時，そのなかの構成要素の和をとったものもDirichlet分布に従う
`$$(X_1, \cdots, X_k + X_{k+1}, \cdots, X_N)
	\sim \mathrm{Dir}(
    \alpha_1, \cdots,
    	\alpha_k + \alpha_{k+1},
        \cdots
    	\alpha_N)$$`
というもの．
これは直感的に説明すれば，LDAのあるトピックを混ぜ合わせて別のトピックを構成してもLDAで扱えることに対応してる．
たとえばセリーグとパリーグに関するトピックがあった時，これを混ぜ合わせると日本のプロ野球のトピックになる．
これを逆にとらえれば，あるトピックを切り分けて別のトピックを作り出せることが分かる．
極端な場合には，無限の切り分けによって無限のトピックを考える事ができる．
この極端な場合をうまく扱う方法論がノンパラベイズであり，
Dirichlet分布をDirichelt Processに置き換えることで対処できる．

###なぜノンパラか？
###過程って何？
###具体的なアルゴリズム
####Stick Breaking Process (SBP)
####Chinese Restaurant Process (CRP)
