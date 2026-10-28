package neuralforge.data;

import neuralforge.TestData;
import java.util.Arrays;
import static neuralforge.TestHelper.*;

public class CsvReaderTest{

    public static void main(String[] args){
        testParseLine();
        testQuotes();
        testParseErrors();
        testReadWithHeader();
        testReadQuirks();
        testReadWithoutHeader();
        testReadErrors();
        finish();
    }

    private static boolean same(String[] a, String... b){
        return Arrays.equals(a, b);
    }

    private static void testParseLine(){
        check("simple", same(CsvReader.parseLine("a,b,c", ',', 1), "a", "b", "c"));
        check("one field", same(CsvReader.parseLine("abc", ',', 1), "abc"));

        check("spaces trimmed", same(CsvReader.parseLine(" a , b ,c ", ',', 1), "a", "b", "c"));

        check("trailing delimiter", same(CsvReader.parseLine("a,b,", ',', 1), "a", "b", ""));
        check("only delimiters", same(CsvReader.parseLine(",,", ',', 1), "", "", ""));

        check("other delimiter", same(CsvReader.parseLine("a;b,c;d", ';', 1), "a", "b,c", "d"));
        check("tab delimiter", same(CsvReader.parseLine("1\t2\tx", '\t', 1), "1", "2", "x"));
    }

    private static void testQuotes(){
        check("comma inside quotes", same(CsvReader.parseLine("\"Smith, John\",42", ',', 1), "Smith, John", "42"));
        check("doubled quote", same(CsvReader.parseLine("\"say \"\"hi\"\"\",x", ',', 1), "say \"hi\"", "x"));
        check("empty quoted", same(CsvReader.parseLine("\"\",x", ',', 1), "", "x"));

        check("quoted spaces kept", same(CsvReader.parseLine("  \"  keep  \"  ,z", ',', 1), "  keep  ", "z"));

        check("quote in the middle", same(CsvReader.parseLine("5\" pipe,x", ',', 1), "5\" pipe", "x"));
    }

    private static void testParseErrors(){
        expectThrows("quote never closed", () -> CsvReader.parseLine("\"abc,def", ',', 1));
        expectThrows("text after closing quote", () -> CsvReader.parseLine("\"abc\"x,1", ',', 1));

        String message = TestData.messageOf(() -> CsvReader.parseLine("\"abc", ',', 7));
        check("error names the line", message != null && message.contains("line 7"));
    }

    private static void testReadWithHeader(){
        String path = TestData.writeText(".csv", "a,b,c\n1,2,3\n4,5,6\n");
        CsvTable table = CsvReader.read(path, true);

        check("column count", table.columnCount() == 3);
        check("row count", table.rowCount() == 2);
        check("header", same(table.getHeader(), "a", "b", "c"));
        check("cell", table.get(1, 2).equals("6"));
        check("line numbers", table.lineNumber(0) == 2 && table.lineNumber(1) == 3);
        check("column index", table.columnIndex("b") == 1);
        expectThrows("unknown column name", () -> table.columnIndex("zzz"));

        table.getHeader()[0] = "changed";
        check("header is a copy", table.getHeader()[0].equals("a"));
    }

    private static void testReadQuirks(){
        String text = "\uFEFFx,y\r\n\r\n1,2\r\n\r\n3,4\r\n\r\n";
        CsvTable table = CsvReader.read(TestData.writeText(".csv", text), true);

        check("bom removed", table.getHeader()[0].equals("x"));
        check("empty lines skipped", table.rowCount() == 2);
        check("line numbers count empty lines", table.lineNumber(0) == 3 && table.lineNumber(1) == 5);
        check("no carriage return left", table.get(0, 1).equals("2"));

        CsvTable semicolons = CsvReader.read(TestData.writeText(".csv", "p;q\n1,5;2\n"), true, ';');
        check("semicolon file", semicolons.get(0, 0).equals("1,5") && semicolons.get(0, 1).equals("2"));
    }

    private static void testReadWithoutHeader(){
        CsvTable table = CsvReader.read(TestData.writeText(".csv", "1,2\n3,4\n"), false);
        check("rows without header", table.rowCount() == 2);
        check("made up names", same(table.getHeader(), "col0", "col1"));
        check("first row is data", table.get(0, 0).equals("1"));
    }

    private static void testReadErrors(){
        String ragged = TestData.writeText(".csv", "a,b,c\n1,2,3\n4,5\n");
        String message = TestData.messageOf(() -> CsvReader.read(ragged, true));
        check("short row is reported with its line", message != null && message.contains("line 3"));

        expectThrows("empty file", () -> CsvReader.read(TestData.writeText(".csv", ""), true));
        expectThrows("only empty lines", () -> CsvReader.read(TestData.writeText(".csv", "\n\n\n"), true));
        expectThrows("file does not exist", () -> CsvReader.read("this/file/does/not/exist.csv", true));
    }
}
