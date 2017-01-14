// Demo the usage of TokensRegx class

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

// For TokensRegx
import edu.stanford.nlp.ling.tokensregex.CoreMapExpressionExtractor;
import edu.stanford.nlp.ling.tokensregex.MatchedExpression;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.ling.tokensregex.TokenSequenceMatcher;

public class TokensRegxDemo
{
    public static void main(String[] args) {
	String[] raw_sentences = {
	    "Picasso is an artist .",
	    "Van Gogh is a painter .",
	    "He is a worker .",
	    "I am a student .",
	    "Picasso is an artist and I am a painter .",
	};

	Properties props = new Properties();
	props.put("annotators", "tokenize, ssplit,  pos, lemma, ner");
	StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	System.out.println("");

	for(int j = 0; j < raw_sentences.length; j++) {
	    String sentStr = raw_sentences[j];
	    System.out.println(sentStr);
	    Annotation document = new Annotation(sentStr);
	    // run all Annotators on this text
	    pipeline.annotate(document);
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    CoreMap sent = sentences.get(0);

	    List<CoreLabel> tokens = sent.get(TokensAnnotation.class);
	    TokenSequencePattern pattern = TokenSequencePattern.compile("([ner:PERSON]+|[pos:PRP]) ([pos:VBZ]|[pos:VBP]) /a|an?/ (/artist|painter|worker|student/)");
	    TokenSequenceMatcher matcher = pattern.getMatcher(tokens);

	    while (matcher.find()) {
		String matchedString = matcher.group(1);
		String job = matcher.group(3);
		List<CoreMap> matchedTokens = matcher.groupNodes();
		System.out.println("Matched person: " + matchedString);
		System.out.println("Matched job: " + job);
	    }
	    System.out.println("");
	}

	System.out.println("TokensRegx over raw strings");
	TokenSequencePattern pattern = TokenSequencePattern.compile("(/He|I|Picasso/) /is|am/ /a|an?/ (/artist|painter|worker|student/)");
	
	for(int j = 0; j < raw_sentences.length; j++) {
	    String sentStr = raw_sentences[j];
	    System.out.println(sentStr);
	    List<CoreLabel> tokens = new ArrayList<CoreLabel>();
	    String[] words = sentStr.split("\\s+");
	    for(int k = 0; k < words.length; k++) {
		CoreLabel tk = new CoreLabel();
		tk.setWord(words[k]);
		tokens.add(tk);
	    }
	    TokenSequenceMatcher matcher = pattern.getMatcher(tokens);
	    while (matcher.find()) {
		String matchedString = matcher.group(1);
		String job = matcher.group(2);
		List<CoreMap> matchedTokens = matcher.groupNodes();
		System.out.println("Matched person: " + matchedString);
		System.out.println("Matched job: " + job);
	    }
	    System.out.println("");
	}
    }
}
