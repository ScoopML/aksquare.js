# aksquare.js # Copper ![test](https://forthebadge.com/images/badges/made-with-javascript.svg) ![test1](https://forthebadge.com/images/badges/built-with-love.svg) [![PyPI download day](https://img.shields.io/pypi/dd/ansicolortags.svg)](https://www.npmjs.com/package/aksquare.js)

A deep learning library built using javascript to build and train machine learning models in the browsers.

### Install
```
npm install aksquare.js --save
```

### Getting Started

via Script tag
Add the following code to an HTML file:

```
<html>
  <head>
  <script>https://cdn.jsdelivr.net/npm/aksquare.js@1.0.1/aksquare/aksquare.min.js</script>

<script>     
let text = "The king is a man who rules over a nation, he always have a woman beside him called the queen.
text_lower = text.toLocaleLowerCase()

text_list = text_lower.split("\n")

var stopwords = ["a","in","when","the","of","is","who"]


let [word_list, all_text] = gen_word(5,text_list);


let unique_dict = unique_word(all_text)

let n_words = obj_len(unique_dict);

console.log(n_words);


let [data, label] = create_data(word_list)


let embed_dim = 50;
let model = new aksquare.Sequential([
        new aksquare.Linear(n_words,embed_dim),
        new aksquare.Linear(embed_dim,n_words),
        new aksquare.Softmax()
]);

let optim = new aksquare.OptimSGD(model,lr=0.001);


epoch = 50
for(let i=0; i< epoch; i++){

    let total_loss = 0;
    for(let j=0; j < data.length; j++){

        let x_data = data[j]
        let y_data = label[j]

        let x = new aksquare.Tensor(1,n_words, false);
        x.setFrom(x_data)

        model.forward(x)

        // console.log(-Math.log(model.out.out[y_data-1]))
        let loss = new aksquare.Loss(y_data-1,model)

        // console.log(loss.out);
        total_loss += loss.out

        loss.backward()

        optim.step();

        optim.grad_zero()

    }

    console.log(`for epoch ${i} Loss is ${total_loss/data.length}`)
}

//get embedding weight

let embed_weight = get_weight(model.models[0].W)
console.log(embed_weight[0].length)

    </script> 
  </head>
 
  <body>
  </body>
</html>

```
Via NPM

Install aksquare.js in your project using yarn or npm

```
import * as aksquare from 'aksquare.js'

let embed_dim = 50;
let model = new aksquare.Sequential([
        new aksquare.Linear(n_words,embed_dim),
        new aksquare.Linear(embed_dim,n_words),
        new aksquare.Softmax()
]);

let optim = new aksquare.OptimSGD(model,lr=0.001);
		let x_data = data[j]
        let y_data = label[j]
	let x = new aksquare.Tensor(1,n_words, false);
        x.setFrom(x_data)

        model.forward(x)

        let loss = new aksquare.Loss(y_data-1,model);
        total_loss += loss.out
		loss.backward()
		optim.step();
		optim.grad_zero()
	}
		let embed_weight = get_weight(model.models[0].W)
console.log(embed_weight[0].length)
```
## For contribution and bugs contact us : harishsg99@gmail.com
