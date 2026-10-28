package neuralforge.data;

import neuralforge.TestData;
import neuralforge.math.Matrix;
import java.util.Arrays;
import static neuralforge.TestData.bytes;
import static neuralforge.TestHelper.*;

public class IdxReaderTest{

    public static void main(String[] args){
        testReadPlain();
        testLimit();
        testGzip();
        testReadImages();
        testReadLabels();
        testBadHeaders();
        testBadSizes();
        testWrongKind();
        finish();
    }

    private static void testReadPlain(){
        String path = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{5}, bytes(1, 2, 3, 4, 5)));
        IdxReader.IdxArray array = IdxReader.read(path, 0);

        check("dims", Arrays.equals(array.getDims(), new int[]{5}));
        check("data", Arrays.equals(array.getData(), bytes(1, 2, 3, 4, 5)));
    }

    private static void testLimit(){
        byte[] data = new byte[16];
        for(int i = 0; i < 16; i++){
            data[i] = (byte) (i + 1);
        }
        String path = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{4, 2, 2}, data));

        IdxReader.IdxArray two = IdxReader.read(path, 2);
        check("limit changes the first dimension", Arrays.equals(two.getDims(), new int[]{2, 2, 2}));
        check("limit loads only the first items", two.getData().length == 8 && two.getData()[7] == 8);

        check("limit 0 loads all", IdxReader.read(path, 0).getData().length == 16);
        check("big limit loads all", IdxReader.read(path, 100).getDims()[0] == 4);
    }

    private static void testGzip(){
        byte[] raw = TestData.idxBytes(new int[]{3}, bytes(7, 8, 9));

        String path = TestData.writeBytes(".bin", TestData.gzip(raw));
        IdxReader.IdxArray array = IdxReader.read(path, 0);
        check("compressed file", Arrays.equals(array.getData(), bytes(7, 8, 9)));

        String plain = TestData.writeBytes(".gz", raw);
        check("plain file with gz name", IdxReader.read(plain, 0).getData().length == 3);
    }

    private static void testReadImages(){
        byte[] pixels = bytes(0, 255, 51, 200, 1, 2, 3, 4);
        String path = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{2, 2, 2}, pixels));

        Matrix images = IdxReader.readImages(path, 0);
        check("shape", images.shape().equals("2x4"));
        check("0 stays 0", images.get(0, 0) == 0.0);
        check("255 becomes 1", images.get(0, 1) == 1.0);
        check("51 becomes 0.2", near(images.get(0, 2), 0.2, 1e-12));
        check("values above 127 are not negative", near(images.get(0, 3), 200.0 / 255.0, 1e-12));
        check("second image", near(images.get(1, 3), 4.0 / 255.0, 1e-12));

        check("images with limit", IdxReader.readImages(path, 1).rows() == 1);
    }

    private static void testReadLabels(){
        String path = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{3}, bytes(0, 9, 200)));
        int[] labels = IdxReader.readLabels(path, 0);
        check("labels", Arrays.equals(labels, new int[]{0, 9, 200}));
        check("labels with limit", IdxReader.readLabels(path, 2).length == 2);
    }

    private static void testBadHeaders(){
        byte[] notIdx = TestData.idxBytes(new int[]{1}, bytes(1));
        notIdx[0] = 1;
        String badMagic = TestData.writeBytes(".idx", notIdx);
        expectThrows("first bytes not zero", () -> IdxReader.read(badMagic, 0));

        String floats = TestData.writeBytes(".idx", TestData.idxBytes(0x0D, new int[]{1}, bytes(0, 0, 0, 0)));
        expectThrows("other value type", () -> IdxReader.read(floats, 0));

        String noDims = TestData.writeBytes(".idx", TestData.idxBytes(new int[0], bytes()));
        expectThrows("zero dimensions", () -> IdxReader.read(noDims, 0));

        String manyDims = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{1, 1, 1, 1, 1}, bytes(1)));
        expectThrows("too many dimensions", () -> IdxReader.read(manyDims, 0));

        String negative = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{-5}, bytes()));
        expectThrows("negative size", () -> IdxReader.read(negative, 0));

        expectThrows("file does not exist", () -> IdxReader.read("no/such/file.idx", 0));
    }

    private static void testBadSizes(){
        String shortFile = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{10}, bytes(1, 2, 3, 4, 5)));
        String message = TestData.messageOf(() -> IdxReader.read(shortFile, 0));
        check("truncated file", message != null && message.contains("too early"));

        String longFile = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{3}, bytes(1, 2, 3, 4, 5)));
        String extra = TestData.messageOf(() -> IdxReader.read(longFile, 0));
        check("extra bytes", extra != null && extra.contains("more data"));

        check("extra bytes are fine with a limit", IdxReader.read(longFile, 2).getData().length == 2);

        String empty = TestData.writeBytes(".idx", new byte[0]);
        expectThrows("empty file", () -> IdxReader.read(empty, 0));

        String huge = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{2, 70000, 70000}, bytes()));
        expectThrows("sizes that overflow", () -> IdxReader.read(huge, 0));

        String tooBig = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{100000, 30000}, bytes()));
        expectThrows("more data than one array can hold", () -> IdxReader.read(tooBig, 0));
    }

    private static void testWrongKind(){
        String labelFile = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{3}, bytes(1, 2, 3)));
        String imageFile = TestData.writeBytes(".idx", TestData.idxBytes(new int[]{1, 2, 2}, bytes(1, 2, 3, 4)));

        expectThrows("label file read as images", () -> IdxReader.readImages(labelFile, 0));
        expectThrows("image file read as labels", () -> IdxReader.readLabels(imageFile, 0));
    }
}
