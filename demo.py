from lib import algorithms as algos

def main():
    analyzed_ast = algos.AnalyzedAST("./tests/rebound.py")

    print('\n')
    print(analyzed_ast.syntax_features)
    print('\n')
    print(analyzed_ast.std_lib_features)
    print('\n')
    print(analyzed_ast.technologies)


if __name__ == "__main__":
    main()
