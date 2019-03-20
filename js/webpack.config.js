const HtmlWebPackPlugin = require("html-webpack-plugin");
// const CopyWebpackPlugin = require('copy-webpack-plugin');
// const MiniCssExtractPlugin = require("mini-css-extract-plugin");

var path = require('path');

var outputPath = path.join(__dirname, 'build');

const tstamp = () => {
  return new Date().toISOString()
      .replace(/\.[\w\W]+?$/, '') // Delete from dot to end.
      .replace(/\:|\s|T/g, '-');  // Replace colons, spaces, and T with hyphen.
};

module.exports = {
  devtool: 'sourcemap',
  resolve: {
    // Add '.ts' and '.tsx' as resolvable extensions.
    extensions: [".ts", ".tsx", ".js", ".json"]
  },
  entry: [
    './src/index.tsx',
  ],
  output: {
    path: outputPath,
    filename: 'main.js', //path.join('static', 'main.js'),
    publicPath: "/static/"
  },
  module: {
    rules: [
      { test: /\.tsx?$/, loader: "awesome-typescript-loader" },
      { enforce: "pre", test: /\.js$/, loader: "source-map-loader" },
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          query: {
            presets: [
              "@babel/preset-env",
              // 'es2015',
              // 'react',
            ],
            plugins: [
              'transform-class-properties',
            ],
          }

        }
      },
      {
        test: /\.html$/,
        use: [
          {
            loader: "html-loader"
          }
        ]
      },
      {
        test: /\.scss$/,
        use: [
          {
            loader: "style-loader" // creates style nodes from JS strings
          },
          {
            loader: "css-loader" // translates CSS into CommonJS
          },
          {
            loader: "sass-loader" // compiles Sass to CSS
          }
        ]
      },
      {
        test: /\.(otf|eot|svg|ttf|woff)/,
        loader: 'url-loader?limit=8192'
      }
    ]
  },
  plugins: [
      new HtmlWebPackPlugin({
        template: "./src/index.html",
        filename: "./index.html",
        hash: true
      }),
  ]

};