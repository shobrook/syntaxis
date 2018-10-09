import React, { Component, Fragment } from "react";
import ReactDOM from "react-dom";
import brace from "brace";
import AceEditor from "react-ace";
import Tree from "react-d3-tree";

import "brace/mode/python";
import "brace/theme/github";

import "./App.scss";

const treeStyle = {
  links: {
    stroke: "#E4E9F3",
    strokeWidth: 2
  },
  nodes: {
    node: {
      circle: {
        fill: "#9EC0FF",
        stroke: "#3E82FF", //"#E4E9F3",
        strokeWidth: 2
      },
      name: {},
      attributes: {
        display: "none"
      }
    },
    leafNode: {
      circle: {
        fill: "#E4E9F3",
        stroke: "#8893AC", //"#E4E9F3",
        strokeWidth: 2
      },
      name: {},
      attributes: {
        display: "none"
      }
    }
  }
};

const demoCode = `from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class Model(object):
\tdef __init__(self, training_data, hyperparameterize=False):
\t\tself.x_train, self.y_train = self.__split(training_data, balanced=True)

\t\tself.scaler = StandardScaler()
\t\tself.scaler.fit(training_data)

\t\tself.model = RandomForestClassifier(n_estimators=500)
\t\tself.model.fit(self.scaler.transform(self.x_train), self.y_train)

\tdef __split(self, df, balanced):
\t\tpass

\tdef predict(self, vector):
\t\treturn self.model.predict(self.scaler.transform(vector))
`;

const parseJSON = (file, callback) => {
  let rawJSON = new XMLHttpRequest();
  rawJSON.overrideMimeType("application/json");
  rawJSON.open("GET", file, true);
  rawJSON.onreadystatechange = () => {
    if (rawJSON.readyState === 4 && rawJSON.status == "200") {
      callback(rawJSON.responseText);
    }
  };

  rawJSON.send(null);
};

class AppView extends Component {
  constructor() {
    super();

    this.state = {
      code: demoCode,
      forest: [],
      showTree: false
    };
  }

  // autoScroll() {
  //   const anchorNode = ReactDOM.findDOMNode(this.refs.anchor)
  //   if (this.state.showTree) {
  //     window.scrollTo(0, anchorNode.offsetTop)
  //   }
  // }

  onSubmit() {
    parseJSON("src/data/tree.json", tree => {
      this.setState({
        forest: [{ name: "", children: JSON.parse(tree) }],
        showTree: !this.state.showTree
      });
    });

    // fetch("https://127.0.0.1:5000/get_pkg_forest", {
    //   mode: "no-cors",
    //   method: "POST",
    //   headers: {
    //     "Accept": "application/json",
    //     "Content-Type": "application/json; charset=UTF-8"
    //   },
    //   body: JSON.stringify({
    //     code: this.state.code
    //   })
    // })
    // .then((response) => { return response.json() })
    // .then((responseData) => this.setState({forest: responseData}))
    // .catch((err) => { console.log(err) })
    //.done()
  }

  render() {
    const { code, showTree, forest } = this.state;

    return (
      <Fragment>
        <div className="main">
          <div className="heading">
            <h1>
              <b>Roots:</b> Python Package Analysis
            </h1>
            <h2>
              Visualize the usage of imported packages in your Python code as a
              parse tree.
            </h2>
          </div>
          <AceEditor
            className="python-editor"
            mode="python"
            theme="github"
            onChange={value => this.setState({ code: value })}
            fontSize={14}
            showPrintMargin={true}
            showGutter={true}
            highlightActiveLine={true}
            value={this.state.code}
            setOptions={{
              enableBasicAutocompletion: false,
              enableLiveAutocompletion: false,
              enableSnippets: false,
              showLineNumbers: true,
              tabSize: 2
            }}
          />
          <a
            className="cta"
            onClick={
              /*() => this.setState({showTree: !showTree})*/ this.onSubmit.bind(
                this
              )
            }
          >
            Extract Trees
          </a>
          {showTree ? (
            <div className="treeWrapper">
              <Tree
                data={this.state.forest}
                styles={treeStyle}
                allowForeignObjects={true}
                translate={{
                  x: 85,
                  y: 195
                }}
                orientation="vertical"
                nodeSvgShape={{
                  shape: "circle",
                  shapeProps: { r: 6 }
                }}
              />
            </div>
          ) : null}
          <div className="divider" />
          <div className="description">
            <div className="what">
              <h1>What is this?</h1>
              <p>
                Roots is a static analysis tool that captures and represents the
                usage of imported packages in a Python program as a parse tree.
                It's part of{" "}
                <a href="https://github.com/shobrook/saplings">saplings</a>, a
                library of algorithms and data structures for working with
                abstract syntax trees in Python.
                <br />
                <br />
                It takes a Python script as input, captures the imported
                packages, and then extracts tree representations of each
                package's lifecycle in the script. This includes all calls and
                sub-calls of functions, objects, and variables related to the
                package. Each node in the tree holds the name of the package
                "feature" and how frequently it's used in the script.
                <br />
                <br />
                The goal of this project is to open the door for more research
                and analysis of common package usage patterns. For example, just
                as <a href="https://mixpanel.com/">Mixpanel</a> analyzes how
                customers use a product, Roots can analyze how developers use a
                library.
              </p>
            </div>
            <div className="how">
              <h1>How it works</h1>
              <p>[Explanation coming soon]</p>
            </div>
          </div>
        </div>
        <footer>
          <span>Made by Jonathan Shobrook</span>
        </footer>
      </Fragment>
    );
  }
}

export default AppView;
