import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author Crunchify.com
 */

public class IterateThroughList {

    public static void main(String[] argv) {

	// create list
	List<String> crunchifyList = new ArrayList<String>();

	// add 4 different values to list
	crunchifyList.add("eBay");
	crunchifyList.add("Paypal");
	crunchifyList.add("Google");
	crunchifyList.add("Yahoo");

	// iterate via "for loop"
	System.out.println("==> For Loop Example.");
	for (int i = 0; i < crunchifyList.size(); i++) {
	    System.out.println(crunchifyList.get(i));
	}

	// iterate via "New way to loop"
	System.out.println("\n==> Advance For Loop Example..");
	for (String temp : crunchifyList) {
	    System.out.println(temp);
	}

	// iterate via "iterator loop"
	System.out.println("\n==> Iterator Example...");
	Iterator<String> crunchifyIterator = crunchifyList.iterator();
	while (crunchifyIterator.hasNext()) {
	    System.out.println(crunchifyIterator.next());
	}

	// iterate via "while loop"
	System.out.println("\n==> While Loop Example....");
	int i = 0;
	while (i < crunchifyList.size()) {
	    System.out.println(crunchifyList.get(i));
	    i++;
	}

	// collection stream() util: Returns a sequential Stream with this collection as its source
	System.out.println("\n==> collection stream() util....");
	crunchifyList.forEach((temp) -> {
		System.out.println(temp);
	    });
    }
}
