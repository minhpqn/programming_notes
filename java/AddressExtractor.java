// Demo address extractors using information from NER

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;

// For TokensRegx
import edu.stanford.nlp.ling.tokensregex.CoreMapExpressionExtractor;
import edu.stanford.nlp.ling.tokensregex.MatchedExpression;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.ling.tokensregex.TokenSequenceMatcher;

public class AddressExtractor {
    public static void main(String[] args) {
	String fileName = "./address_with_ner.txt";
	Pattern r = Pattern.compile("^output: (.+)$");
	Pattern t1 = Pattern.compile("<([^/]+)>(.+)</(.+)>(,?)");
	Pattern t2 = Pattern.compile("<([^/]+?)>(.+)");
	Pattern t3 = Pattern.compile("(.+)</(.+)>(,?)");

	TokenSequencePattern pattern = TokenSequencePattern.compile("([ner:ADDRESS]* /,*/)* ([ner:DISTRICT]* /,*/)* ([ner:PROVINCE]|[ner:DISTRICT])*");

	try(BufferedReader is = new BufferedReader(new FileReader(fileName))) {
	    String line;
	    
	    while ( (line = is.readLine()) != null ) {
		Matcher m = r.matcher(line);
		if ( m.find() ) {
		    List<CoreLabel> tokens = new ArrayList<CoreLabel>();
		    
		    String text = m.group(1);
		    String[] words = text.split("\\s+");
		    int i = 0;
		    while ( i < words.length ) {
			CoreLabel tk = new CoreLabel();
			String ner  = "";
			String word = words[i];
			Matcher m1 = t1.matcher(words[i]);
			Matcher m2 = t2.matcher(words[i]);
			if (m1.find()) {
			    ner  = m1.group(1);
			    word = m1.group(2);
			    String c = m1.group(4);
			    tk.setWord(word);
			    tk.setNER(ner);
			    tokens.add(tk);
			    if (c.equals(",")) {
				CoreLabel _tk = new CoreLabel();
				_tk.setWord(",");
				_tk.setNER("");
				tokens.add(_tk);
			    }
			    i++;
			}
			else if (m2.find())  {
			    ner = m2.group(1);
			    List<String> wl = new ArrayList<String>();
			    wl.add(m2.group(2));
			    
			    // System.out.format("%d %s\n", i, words[i]);
			    int j = i;
			    while ( j < words.length && ! t3.matcher(words[j]).find() ) {
				j++;				
			    }
			    
			    if ( j == words.length ) {
				j = i;
			    }
			    
			    //System.out.format("%d %s\n", j, words[j]);
			    for ( int k = i+1; k < j; k++ ) {
				wl.add(words[k]);
			    }
			    
			    Matcher m3 = t3.matcher(words[j]);
			    String c = "";
			    if ( m3.find() ) {
				wl.add(m3.group(1));
				if ( m3.group(3).equals(",") ) {
				    //System.out.println(m3.group(3));
				    c = ",";
				}
			    }
			    
			    for ( int k = 0; k < wl.size(); k++ ) {
				CoreLabel _tk = new CoreLabel();
				_tk.setWord(wl.get(k));
				_tk.setNER(ner);
				tokens.add(_tk);
			    }

			    if ( c.equals(",") ) {
				CoreLabel _tk = new CoreLabel();
				_tk.setWord(",");
				_tk.setNER("");
				tokens.add(_tk);
			    }
			    i = ++j;
			}
			else {
			    tk.setWord(word);
			    tk.setNER(ner);
			    tokens.add(tk);
			    i++;
			}
			
		    }
		    // Add code to match address patternp
		    TokenSequenceMatcher matcher = pattern.getMatcher(tokens);
		    String senStr = getSenStr(tokens);
		    System.out.format("Input tokens: %s\n", senStr);
		    while (matcher.find()) {
		    	String matchedString = matcher.group();
		    	System.out.println("Address: " + matchedString);
		    }
		    System.out.println("");
		    
		    // for (i = 0; i < tokens.size(); i++) {
		    // 	CoreLabel token = tokens.get(i);
		    // 	// this is the text of the token
		    // 	String word = token.get(TextAnnotation.class);
		    // 	// this is the NER label of the token
		    // 	String ne = token.get(NamedEntityTagAnnotation.class);
		    // 	System.out.println("word: " + word  + " ne: " + ne);
		    // }
		}
	    }
	} catch(IOException e) {
	    e.printStackTrace();
	}
    }

    private static String getSenStr(List<CoreLabel> tokens) {
	List<String> strings = new ArrayList<String>();
	for(int i = 0; i < tokens.size(); i++) {
	    CoreLabel tk = tokens.get(i);
	    strings.add(tk.get(TextAnnotation.class));
	}
	String senStr = String.join(" ", strings);
	
	return senStr;
    }
}
