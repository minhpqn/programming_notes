// Demo the usage of Stanford CoreNLP package

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class CorenlpDemo {
    public static void main(String[] args) {
	String[] raw_sentences = {
	    "Picasso is an artist.",
	    "Van Gogh is a painter.",
	    "He is a worker.",
	};

	Properties props = new Properties();
	props.put("annotators", "tokenize, ssplit,  pos, lemma, ner");
	StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    
	for(int j = 0; j < raw_sentences.length; j++) {
	    String sentStr = raw_sentences[j];
	    System.out.println(sentStr);
	    Annotation document = new Annotation(sentStr);
	    // run all Annotators on this text
	    pipeline.annotate(document);

	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);

	    for(CoreMap sentence: sentences) {
		// traversing the words in the current sentence
		// a CoreLabel is a CoreMap with additional token-specific methods
		for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
		    // this is the text of the token
		    String word = token.get(TextAnnotation.class);
		    // this is the POS tag of the token
		    String pos = token.get(PartOfSpeechAnnotation.class);
		    // this is the NER label of the token
		    String ne = token.get(NamedEntityTagAnnotation.class);

		    System.out.println("word: " + word + " pos: " + pos + " ne:" + ne);
		}
	    }
	}
    }
}

